# 16-825 Assignment 1: Rendering Basics with PyTorch3D (Total: 100 Points + 10 Bonus)

## Usage Instructions

The `main.py` file can be used to run for all the questions in this assignment

For question 1
```
python main.py --question q1
```

For question 2
```
python main.py --question q2
```

For question 3
```
python main.py --question q3
```

For question 4
```
python main.py --question q4
```

For question 5
```
python main.py --question q5 --render rgbd
python main.py --question q5 --render parametric
python main.py --question q5 --render implicit
```

For question 6
```
python main.py --question q6
```

To generate variations, feel free to modify the `CONFIG_DICT` at the top of `main.py` file

## Known Issues
You may run out of GPU memory when trying to render more than 36 views for q5 (parameteric), since I am not using a for loop and rendering them together as a batch
