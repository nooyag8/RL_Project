import gymnasium as gym
import numpy as np

from envs.domain_randomization import DomainRandomizationWrapper

EXPERIMENTS = {
    "B: Mass DR": (True, False),
    "C: Damping DR": (False, True),
    "D: Combined DR": (True, True),
}

for name, (mass_dr, damping_dr) in EXPERIMENTS.items():

    print(f"\n========== {name} ==========")

    env = DomainRandomizationWrapper(
        gym.make("Reacher-v5"), randomize_mass=mass_dr, randomize_damping=damping_dr
    )

    try:
        for episode in range(3):

            obs, info = env.reset()

            mass = env.model.body_mass[env.body_ids].copy()
            damping = env.model.dof_damping[env.dof_ids].copy()

            mass_scale = mass / env.original_mass[env.body_ids]
            damping_scale = damping / env.original_damping[env.dof_ids]

            print(f"\nEpisode {episode + 1}")
            print("Mass Scale:", mass_scale)
            print("Damping Scale:", damping_scale)

            if mass_dr:
                assert np.all((mass_scale >= 0.8) & (mass_scale <= 1.2))
            else:
                assert np.allclose(mass_scale, 1.0)

            if damping_dr:
                assert np.all((damping_scale >= 0.5) & (damping_scale <= 1.5))
            else:
                assert np.allclose(damping_scale, 1.0)

            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

        print(f"\n{name}: TEST PASSED")

    finally:
        env.close()

print("\nALL TESTS PASSED")
