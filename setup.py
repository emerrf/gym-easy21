from setuptools import setup

setup(
    name='gym_easy21',
    version='1.0.0',
    description='OpenAI Gym environment of the RL Assignment Easy21 game',
    url='',
    author='Emer Rodriguez Formisano',
    author_email='',
    license='GNU General Public License v3.0',
    install_requires=[
        'gym>=0.8.1',
        'numpy==1.12.1',
        'matplotlib==2.0.0'
    ],
    entry_points={
        'console_scripts': [
            'gym-easy21-run=gym_easy21.envs.easy21_run:run',
        ]
    }
)