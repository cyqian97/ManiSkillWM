python examples/baselines/ppo/ppo_rgb.py --env_id="MyTestEnv-v0" \
   --evaluate --checkpoint="runs/MyTestEnv-v0__ppo_rgb__1__1764821918/final_ckpt.pt" \
  --control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos" \
   --num_eval_envs=1 --num-eval-steps=1000