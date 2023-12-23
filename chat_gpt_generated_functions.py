from dm_control import composer
from dm_control.composer.variation import distributions
from dm_control.locomotion.arenas import corridors as corr_arenas
from dm_control.locomotion.tasks import corridors as corr_tasks
from dm_control.locomotion.walkers import cmu_humanoid


def cmu_humanoid_run_gaps_step1(random_state=None, gap_lengths=(1, 2)):
    walker = cmu_humanoid.CMUHumanoidPositionControlled()
    arena = corr_arenas.GapsCorridor(
        platform_length=distributions.Uniform(0.3, 1.5),  # Shorter platforms
        gap_length=distributions.Uniform(gap_lengths[0], gap_lengths[1]),
        corridor_width=10,
        corridor_length=50,  # Shorter corridor
    )
    task = corr_tasks.RunThroughCorridor(
        walker=walker,
        arena=arena,
        walker_spawn_position=(0.5, 0, 0),
        target_velocity=2.0,  # Slower velocity
        physics_timestep=0.005,
        control_timestep=0.03,
    )
    return composer.Environment(
        time_limit=20,  # Shorter time limit
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )


def cmu_humanoid_run_gaps_step2(random_state=None, gap_lengths=(1, 2)):
    walker = cmu_humanoid.CMUHumanoidPositionControlled()
    arena = corr_arenas.GapsCorridor(
        platform_length=distributions.Uniform(0.3, 1.5),
        gap_length=distributions.Uniform(gap_lengths[0], gap_lengths[1]),
        corridor_width=10,
        corridor_length=75,  # Slightly longer corridor
    )
    task = corr_tasks.RunThroughCorridor(
        walker=walker,
        arena=arena,
        walker_spawn_position=(0.5, 0, 0),
        target_velocity=2.5,  # Moderate velocity
        physics_timestep=0.005,
        control_timestep=0.03,
    )
    return composer.Environment(
        time_limit=25,  # Moderately longer time limit
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )


def cmu_humanoid_run_gaps_step3(random_state=None, gap_lengths=(1, 3)):
    walker = cmu_humanoid.CMUHumanoidPositionControlled()
    arena = corr_arenas.GapsCorridor(
        platform_length=distributions.Uniform(0.3, 2.0),  # Moderate platform length
        gap_length=distributions.Uniform(gap_lengths[0], gap_lengths[1]),
        corridor_width=10,
        corridor_length=100,  # Original corridor length
    )
    task = corr_tasks.RunThroughCorridor(
        walker=walker,
        arena=arena,
        walker_spawn_position=(0.5, 0, 0),
        target_velocity=3.5,  # Faster velocity
        physics_timestep=0.005,
        control_timestep=0.03,
    )
    return composer.Environment(
        time_limit=30,  # Original time limit
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )


def cmu_humanoid_run_gaps_step4(random_state=None, gap_lengths=(2, 4)):
    walker = cmu_humanoid.CMUHumanoidPositionControlled()
    arena = corr_arenas.GapsCorridor(
        platform_length=distributions.Uniform(0.5, 2.5),  # Longer platforms
        gap_length=distributions.Uniform(gap_lengths[0], gap_lengths[1]),
        corridor_width=10,
        corridor_length=150,  # Longer corridor
    )
    task = corr_tasks.RunThroughCorridor(
        walker=walker,
        arena=arena,
        walker_spawn_position=(0.5, 0, 0),
        target_velocity=4.0,  # Challenging velocity
        physics_timestep=0.005,
        control_timestep=0.03,
    )
    return composer.Environment(
        time_limit=40,  # Longer time limit
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )
