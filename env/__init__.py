from gym.envs.registration import register

register(
    id='timetable-v0001',
    entry_point='env.env:TimeTable'
)