import gymnasium as gym
import mujoco
import numpy as np


class DomainRandomizationWrapper(gym.Wrapper):

    def __init__(
        self,
        env,
        randomize_mass=False,
        randomize_damping=False,
        mass_range=(0.8, 1.2),
        damping_range=(0.5, 1.5),
    ):
        super().__init__(env)

        self.base_env = env.unwrapped
        self.model = self.base_env.model
        self.data = self.base_env.data

        self.randomize_mass = randomize_mass
        self.randomize_damping = randomize_damping

        self.mass_range = mass_range
        self.damping_range = damping_range

        # 난수 생성기 초기화
        self.rng = np.random.default_rng()

        # 원래 물리 파라미터 저장
        self.original_mass = self.model.body_mass.copy()
        self.original_inertia = self.model.body_inertia.copy()
        self.original_damping = self.model.dof_damping.copy()

        # 변경 대상 Body ID 찾기
        self.body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ["body0", "body1"]
        ]

        # 변경 대상 Joint ID 찾기
        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ["joint0", "joint1"]
        ]

        # 모델 이름 검증
        if any(body_id < 0 for body_id in self.body_ids):
            raise ValueError("Reacher body name not found.")

        if any(joint_id < 0 for joint_id in self.joint_ids):
            raise ValueError("Reacher joint name not found.")

        # Joint ID를 DOF 주소로 변환
        self.dof_ids = [
            int(self.model.jnt_dofadr[joint_id]) for joint_id in self.joint_ids
        ]

    def reset(self, *, seed=None, options=None):

        # Seed 설정
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # 원래 파라미터 복구
        self.model.body_mass[:] = self.original_mass
        self.model.body_inertia[:] = self.original_inertia
        self.model.dof_damping[:] = self.original_damping

        # Mass Randomization
        if self.randomize_mass:

            mass_scale = self.rng.uniform(*self.mass_range)

            for body_id in self.body_ids:

                self.model.body_mass[body_id] = self.original_mass[body_id] * mass_scale

                self.model.body_inertia[body_id] = (
                    self.original_inertia[body_id] * mass_scale
                )

        # Damping Randomization
        if self.randomize_damping:

            damping_scale = self.rng.uniform(*self.damping_range)

            for dof_id in self.dof_ids:

                self.model.dof_damping[dof_id] = (
                    self.original_damping[dof_id] * damping_scale
                )

        # MuJoCo 파생 상수 갱신
        mujoco.mj_setConst(self.model, self.data)

        # 환경 초기화
        return self.env.reset(seed=seed, options=options)
