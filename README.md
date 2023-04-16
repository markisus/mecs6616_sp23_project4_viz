Simulator / Visualizer for running Robotic Learning HW4 Locally
Forked from https://github.com/roamlab/mecs6616_sp23_project4.git

You will need to locally pip install any dependencies from the original repo.  
Additionally you may need to run
`pip install imgui[sdl2]`.

On Windows you will also need
`pip install pysdl2-dll`.  


Run `main.py`. To use student dynamics, add argument `--use-student`.

Copy your MPC into `mpc.py`.
Copy your student dynamics into `arm_student.py`.

Additionally there is a dataset viewer tool for your training data at `data_viewer.py`. It expects the pkl file at `dataset/data.pkl`








