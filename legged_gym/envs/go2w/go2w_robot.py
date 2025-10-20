from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os  

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain  
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, get_scale_shift
from legged_gym.utils.helpers import class_to_dict
from .go2w_amp_config import GO2WAMPCfg
from rsl_rl.datasets.motion_loader import AMPLoader

class Go2w(LeggedRobot):
    cfg: GO2WAMPCfg

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        
        # self.obs_hist_buf = self.obs_hist_buf[:,73:]
        # self.obs_hist_buf = torch.cat((self.obs_hist_buf,self.obs_buf),dim = -1)
        # self.prev_privileged_obs_buf = self.privileged_obs_buf


        clip_actions = self.cfg.normalization.clip_actions # 将动作限制在clip_actions参数范围内
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render() # 渲染环境，可视化
        for _ in range(self.cfg.control.decimation): # 在每个控制周期中进行多少次物理步进
            self.torques = self._compute_torques(self.actions).view(self.torques.shape) 
            # 根据action计算扭矩
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            # 将计算得到的扭矩应用到仿真环境中的关节上。
            self.gym.simulate(self.sim) # 更新环境状态
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)  
        reset_env_ids, terminal_amp_states = self.post_physics_step() # 物理仿真步进后额外处理

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        # 裁剪观测值
        
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, reset_env_ids, terminal_amp_states ## do we need to return obs history buffer??
        # return policy_obs, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, reset_env_ids, terminal_amp_states

    def compute_observations(self):
        """ Computes observations
        """
        # Blind, no perceptive observation
        # print('计算')
        #self._local_gripper_pos = torch.zeros((self.num_envs,3),dtype=torch.float,device=self.device)
        self.base_height_command = torch.tensor(self.cfg.rewards.base_height_target, dtype=torch.float, device=self.device)
        self.base_height_command = self.base_height_command.unsqueeze(0).repeat(self.num_envs,1)

        self.dof_err = self.dof_pos - self.default_dof_pos # 机器人当前各DOF位置 - 机器人默认各DOF位置 看作一种位置的偏差值
        self.dof_err[:,self.wheel_indices] = 0 # 轮子的位置偏差值设置为0，因为轮子在每个位置都是一样的
        self.dof_pos[:,self.wheel_indices] = 0 # 轮子的关节位置设置为0，理由同上
        self.obs_buf = torch.cat((  #self.base_lin_vel * self.obs_scales.lin_vel, # 机器人坐标下的基座线速度 * 2.0
                                    self.base_ang_vel  * self.obs_scales.ang_vel, # 机器人坐标下的基座角速度 * 0.25
                                    self.projected_gravity, # 机器人坐标系下重力向量 
                                    self.commands[:, :3] * self.commands_scale, 
                                    # X轴线速度 Y轴线速度 角速度指令
                                    # 外部控制命令 * [lin_vel, lin_vel, ang_vel] = [2.0, 2.0, 0.25]
                                    # self.base_height_command, # 基座机器人高度命令 
                                    # 在训练中command命令是一段采样时间后随机生成的
                                    self.dof_err * self.obs_scales.dof_pos, # 关节误差 * 1.0
                                    self.dof_vel * self.obs_scales.dof_vel, # 机器人各DOF速度 * 0.05
                                    self.dof_pos, # 机器人各DOF位置
                                    self.actions, # 动作输入
                                    ),dim=-1)
        
        # lin_vel = 2.0 ang_vel = 0.25 dof_pos = 1.0 dof_vel = 0.05 height_measurements = 5.0 clip_observations = 100. clip_actions = 100.
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            # print('有噪声')
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec # 添加噪声
            # print(self.obs_buf.size())

        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
        
        contact_forces_scale, contact_forces_shift = get_scale_shift(self.cfg.normalization.contact_force_range)
        
        # 计算特权信息
        self.privileged_obs_buf = torch.cat(( self.obs_buf,
                                             self.base_lin_vel*self.obs_scales.lin_vel,
                                             (self.contact_forces.view(self.num_envs, -1) - contact_forces_shift) * contact_forces_scale,
                                             heights
                                             ),dim=-1)
        # print(self.privileged_obs_buf.shape, (self.contact_forces.view(self.num_envs, -1) - contact_forces_shift).shape, heights.shape, self.measured_heights, '######################')

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
    
    def post_physics_step(self):
        """ check terminations, compute observations and rewards  检查是否终止训练、计算观测值和奖励
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        # 表示机器人基座的状态
        # 3个浮点数表示位置，4个浮点数表示四元数，3个浮点数表示线速度，3个浮点数表示角速度
        # (num_actors, 13)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # 表示机器人每一部分的状态
        # 3个浮点数表示位置，4个浮点数表示四元数，3个浮点数表示线速度，3个浮点数表示角速度
        # (num_actors, num_rigid_bodys, 13)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # 表示机器人每个DOF的状态
        # 多个DOF，每个DOF状态由两个浮点数表示：DOF位置和DOF速度
        # 平移DOF：m,m/s; 旋转DOF：弧度，弧度/s
        # (num_actors, num_dofs, 2) 
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        # 包含每个刚体在最后一个模拟步骤中经历的净接触力
        # 数据都是按顺序排列，首先是actor1的所有信息，然后是actor2的所有信息，巴拉巴拉……

        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # 作用是实时获取最新的以上这些信息
        
        self.root_states = gymtorch.wrap_tensor(actor_root_state).view(-1, 13)
        # 机器人基座信息：3个浮点数表示位置，4个浮点数表示四元数，3个浮点数表示线速度，3个浮点数表示角速度
        
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        # 机器人各部分刚体信息

        self.base_handles = self.gym.find_asset_rigid_body_index(self.robot_asset, "base_link")
        self.base_pos = self.rigid_body_states[:, self.base_handles][:, 0:3]
        # 这部分是那篇论文中的将基座和基座上固定的机械手的信息拿取出来
        
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0] # 各DOF位置信息
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1] # 各DOF速度信息
        self.base_quat = self.root_states[:, 3:7] # 机器人基座四元数信息
        # 这个应该还是获得机械抓手的位置的
        #get local frame quat
        #self.local_frame_quat = 

        
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        
        assert not torch.isnan(self.contact_forces).any(), "privileged_obs contains NaN values"
        assert not torch.isinf(self.contact_forces).any(), "privileged_obs contains Inf values"
        
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7] # 机器人基座四元数信息
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        # 根据四元数信息将机器人在世界坐标系中的线速度、角速度、重力向量转换为机器人坐标系

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward() # 计算奖励
        #print("**self.reset_buf_buf:",self.reset_buf)
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        terminal_amp_states = self.get_amp_observations()[env_ids]
        
        self.reset_idx(env_ids)
        
        #update_command 
        
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]  # 保留上一时刻action、所有DOF速度、机器人基座线速度、角速度

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
        
        return env_ids, terminal_amp_states

    def get_amp_observations(self):
        joint_indices = torch.tensor([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14], device=self.dof_pos.device)
        
        joint_pos = self.dof_pos[:, joint_indices]
        # foot_pos = self.foot_positions_in_base_frame(self.dof_pos).to(self.device)
        base_lin_vel = self.base_lin_vel
        base_ang_vel = self.base_ang_vel
        joint_vel = self.dof_vel[:, joint_indices]
        # z_pos = self.root_states[:, 2:3]
        return torch.cat((joint_pos, base_lin_vel, base_ang_vel, joint_vel), dim=-1)
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:28] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[28:44] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[44:60] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[60:247] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec
        
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """    
        # 输出力矩用
        # self.log_dir = "./logs"  # 日志文件夹路径
        # if not os.path.exists(self.log_dir):  
        #     os.makedirs(self.log_dir)
        # self.log_file = os.path.join(self.log_dir, "torques.log")
        #pd controller
        dof_err = self.default_dof_pos - self.dof_pos # 各DOF默认位置 - 目前各DOF位置
        dof_err[:,self.wheel_indices] =  0 # 轮子的误差是0
        actions_scaled = actions * self.cfg.control.action_scale # action * 0.25
        actions_scaled[:, self.wheel_indices] = 0 # 轮子使用速度控制，角度增量为0
        vel_ref = torch.zeros_like(actions_scaled)
        vel_tmp = actions * self.cfg.control.vel_scale # action提供期望速度
        vel_ref[:, self.wheel_indices] = vel_tmp[:, self.wheel_indices] # 只有轮子使用速度控制
        control_type = self.cfg.control.control_type # 其实就是P
        if control_type=="P":
            torques = self.p_gains * (
                actions_scaled + dof_err
            ) + self.d_gains * (vel_ref - self.dof_vel)
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        # 输出力矩用
        # with open(self.log_file, "a") as f:  # 追加模式写入
        #     f.write(f"{torques.tolist()}\n")  # 将tensor转换为list再写入
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    
    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.robot_asset = robot_asset
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
        wheel_names =[]
        for name in self.cfg.asset.wheel_name:
            wheel_names.extend([s for s in self.dof_names if name in s])
        print("###self.rigid_body names:",body_names)
        print("###self.dof names:",self.dof_names)
        print("###penalized_contact_names:",penalized_contact_names)
        print("###termination_contact_names:",termination_contact_names)
        print("###feet_names:",feet_names)
        print("###wheels name:",wheel_names)
        
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            print("termination_contact_names[i]",termination_contact_names[i])
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
            print("termination_contact_names[i]",termination_contact_names[i],"indice[i]:",self.termination_contact_indices[i])
        
        self.wheel_indices = torch.zeros(len(wheel_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(wheel_names)):
            self.wheel_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], wheel_names[i])

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        if self.cfg.env.reference_state_initialization:
            frames = self.amp_loader.get_full_frame_batch(len(env_ids))
            self._reset_dofs_amp(env_ids, frames)
            # print(env_ids, frames)
            self._reset_root_states_amp(env_ids, frames)
        else:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        if self.cfg.domain_rand.randomize_gains:
            new_randomized_gains = self.compute_randomized_gains(len(env_ids))
            self.randomized_p_gains[env_ids] = new_randomized_gains[0]
            self.randomized_d_gains[env_ids] = new_randomized_gains[1]

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _reset_dofs_amp(self, env_ids, frames):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
            frames: AMP frames to initialize motion with
        """
        joint_indices = torch.tensor([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14], device=self.dof_pos.device)
        
        self.dof_pos[env_ids.unsqueeze(1), joint_indices.unsqueeze(0)]= AMPLoader.get_joint_pose_batch(frames)
        self.dof_vel[env_ids.unsqueeze(1), joint_indices.unsqueeze(0)] = AMPLoader.get_joint_vel_batch(frames)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states_amp(self, env_ids, frames):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        root_pos = AMPLoader.get_root_pos_batch(frames)
        
        root_pos[:, :2] = root_pos[:, :2] + self.env_origins[env_ids, :2]
        self.root_states[env_ids, :3] = root_pos
        root_orn = AMPLoader.get_root_rot_batch(frames)
        self.root_states[env_ids, 3:7] = root_orn

        self.root_states[env_ids, 7:10] = quat_rotate(root_orn, AMPLoader.get_linear_vel_batch(frames))
        self.root_states[env_ids, 10:13] = quat_rotate(root_orn, AMPLoader.get_angular_vel_batch(frames))

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    #==================== reward function ======================================
    def _reward_stand_still(self):
        # Penalize motion at zero commands        
        dof_err = self.dof_pos - self.default_dof_pos
        dof_err[:,self.wheel_indices] = 0
        return torch.sum(torch.abs(dof_err), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        self.dof_vel[:,self.wheel_indices] = 0
        return torch.sum(torch.square(self.dof_vel), dim=1)