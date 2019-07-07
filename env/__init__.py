from gym.envs.registration import register

register(
    id='timetable-v0001',
    entry_point='env.env:TimeTable'
)
register(
    id='timetable-case0001-v0001',
    entry_point='env.case0001:TimeTable'
)