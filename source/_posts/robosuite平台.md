---
title: robosuite平台
tags:
  - robotic
  - simulation
mathjax: false
categories: it
date: 2024-12-02 14:22:27
---


机器人模拟平台robosuite ，最后还是不用了。

<!--more-->

[github主页](https://github.com/ARISE-Initiative/robosuite)

# introduction
## overview
robosuite是一个由MuJoCo物理引擎支持的机器人学习仿真框架。它还提供了一套可重复研究的基准环境。当前版本支持各种机器人实施例（包括类人机器人），自定义机器人组合，复合控制器（包括全身控制器），更多的远程操作设备，照片般逼真的渲染。

该项目是更广泛的通过模拟环境推进机器人智能计划的一部分，旨在降低人工智能和机器人交叉领域尖端研究的进入门槛。

教程部分参考[robosuite.ai](https://robosuite.ai/docs/installation.html)

## install
### windows
主要参考：https://robosuite.ai/docs/installation.html

**但是，pip的跟最新的不一样，建议走source安装法**


#### pre
按官网流程控制台好几次都出现modulenotfound的问题，现在换了idea配置conda，参考[Idea配置anaconda](https://blog.csdn.net/tqlisno1/article/details/108908775)，具体的就是选择3.10的python

不知道管不管用，这次我重启了之后还正常，看看明天怎么样（怎么玄学起来了）

-----

1. 前提：安装好anaconda
2. 环境：`conda create -n robosuite python=3.10`
3. use：`conda activate robo`


#### install from source
1. `git clone https://github.com/ARISE-Initiative/robosuite.git`
   
   `cd robosuite`
2. `pip3 install -r requirements.txt`
3. `pip3 install -r requirements-extra.txt`
4. 测试：`python -m robosuite.demos.demo_random_action`
5. 可能会遇到问题，我遇到的是mujoco.dll not found，参考教程
   1. 先找到MuJoCo安装目录
      1. 保存
         ```python
            import mujoco
            print(mujoco.__path__)
         ```
      2. 在robosuite环境下执行
      3. 得到`D:\\conf\\anaconda\\envs\\robosuite\\lib\\site-packages\\mujoco`
   2. 把dll文件copy and paste this file into anaconda3\envs\{your env name}\Lib\site-packages\robosuite\utils
6. 其它问题翻文档

jianyi 
后面在使用之前都要启动环境`conda activate robosuite`

跑的时候如果出现'ModuleNotFound'，多半是系统路径的问题，因为之前用pip intall郭robosuite，里面代码跑的时候会优先找anaconda里面的robosuite而不是项目的，所以要把之前残留的删了（只pip uninstall没用，也可能是因为我没重启）

### linux
1. linux版本：`uname -a` 后输出Linux qwe-NUC11TNKi5 6.8.0-49-generic #49~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Wed Nov  6 17:42:15 UTC 2 x86_64 x86_64 x86_64 GNU/Linux
2. 安装mmujoco
   1. 先检查以前的，应该是统一在～目录下，如果有.mujoco文件夹就是以前装过，一看是mujoco210，把从它给删了，`sudo rm -rf mujoco210`
   2. 下载mujoco200：`wget https://roboti.us/download/mujoco200_linux.zip`
   3. 解压并将所得文件夹命名为mujoco200
      1. sudo mkdir mujoco200
      2. sudo unzip mujoco200_linux.zip 
      3. sudo mv mujoco200_linux mujoco200
      4. ls检查一下，确保.mujoco里只有mujoco200，进取之后就是bin、include那些
   4. mjkey
      1. wget https://roboti.us/file/mjkey.txt ，下载建议在.mujoco文件夹下
      2. mjkey要放在/mujoco200 和 /mujoco200/bin 这两个文件夹下
      3. sudo cp mjkey.txt .mujoco/
      4. sudo cp mjkey.txt .mujoco/mujoco200/bin/
      5. crtl+H可以查看隐藏的文件，如果有图形页面的话就这么检查，没有的话cd+ls检查下
   5. 配置bashrc
      1. sudo vim ~/.bashrc
      2. i
      3. 最后加上`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin`
      4. :wq!
      5. source ~/.bashrc
   6. 验证
      1. 在 ~/.mujoco/mujoco200/bin$ 执行`./simulate ../model/humanoid.xml `
      2. 能弹出窗口就算成功 
3. mujoco-py 这里会很复杂 失败:( 没能安装成功
   1. sudo apt-get update
   2. sudo apt-get install patchelf
   3. sudo apt-get install python3 python-dev python3-dev build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev libglew1.5 libglew-dev python3-pip 报错不用管
   4. sudo apt-get install python3-pip
   5. git clone https://github.com/openai/mujoco-py
   6. cd mujoco-py
   7. pip3 install -e . --no-cache
   8. cd
   9. 处理第三步报错
      1. 在bashrc开头加`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/qwe/.mujoco/mujoco200/bin`，qwe是用户名字，教程是加在119行，conda之前
      2. source ~/.bashrc
   10. sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
   11. sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
   12. pip3 install -U 'mujoco-py<2.1,>=2.0'这一步没有问题的话就好了
       1.  报错Cython.Compiler.Errors.CompileError: /tmp/pip-install-pubnu2cw/mujoco-py_88165ca0dbf64d608cb465d9e21422e4/mujoco_py/cymj.pyx
       2.  尝试pip install Cython==3.0.0a10不行
       3.  卡住了orz，检查了是安装GraalPy时出现的问题

4. 从github按扎ungrobosuite
   1. conda create -n robosuite python=3.10
   2. conda activate robosuite
   3. git clone https://github.com/ARISE-Initiative/robosuite.git
      1. 或者git clone git@github.com:ARISE-Initiative/robosuite.git
   4. cd robosuite
   5. pip3 install -r requirements.txt
   6. pip3 install -r requirements-extra.txt
   7. python robosuite/demos/demo_random_action.py
      1. 出现libGL error: failed to load driver: swrast
         /home/qwe/anaconda3/envs/robosuite/lib/python3.10/site-packages/glfw/__init__.py:917: GLFWError: (65543) b'GLX: Failed to create context: BadValue (integer parameter out of range for operation)'
         warnings.warn(message, GLFWError)
         ERROR: could not create window
      2.  conda install -c conda-forge gcc

（第一次指导原来{[]}这个键可以切换视角，好鬼畜阿








## basic usage
主要参考：https://robosuite.ai/docs/basicusage.html

```python
import numpy as np
import robosuite as suite

# create environment instance
env = suite.make(
    env_name="Lift", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# reset the environment
env.reset()

for i in range(1000):
    action = np.random.randn(*env.action_spec[0].shape) * 0.1
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display
```
上面的脚本使用屏幕渲染器创建了一个模拟环境，这对于可视化和定性评估很有用。step（）函数接受一个动作作为输入，并返回一个元组（obs，reward，done，info），其中obs是一个包含观测值[（name_string，np.array），.]的OrderedDict，reward是每步获得的即时奖励，done是指示剧集是否已终止的布尔标志，info是包含附加元数据的字典。

## demo
从[demo](https://github.com/ARISE-Initiative/robosuite/tree/master/robosuite/demos)里面下载demo

1. demo_random_action.py 随机动作
2. demo_control.py 需要pip install robosuite_models，演示了robosuite中每个控制器的各种功能，这个没咋看懂，controller是啥
3. demo_domain_randomization.py 需要mujoco-py version 3.1.1 不知道版本的话用`print(mujoco.__version__)`，需要先卸载之前的，再根据版本下载`pip install mujoco==3.1.1` ，域随机化
4. demo_sensor_corruption.py 传感器 为啥这个可以操控了，输出挺恐怖的
5. demo_gripper_selection.py 展示了能用的夹子
6. demo_gripper_interaction.py 我怎么没找到交互的地方……将夹持器导入场景并使其与具有执行器的对象交互的过程。它还展示了如何使用MJCF实用程序函数的Modeling API按程序生成场景
7. demo_collect_and_playback_data.py 轨迹回放
8. demo_gym_functionality.py 需要先安装gymnasium，如何调整环境以兼容OpenAI Gym-style API，基于OpenAI gym文档Getting Started with Gym部分中的一些代码片段编写此脚本。可以把自己的数据集封装到gym中
9.  demo_device_control.py 终于可以控制设备了，参数--device，可以选择Keyboard或者spacemouse，但是spacemouse我没有，目前仅支持mac和linux
   1.  Ctrl+q	reset simulation
   2.  spacebar	toggle gripper (open/close)
   3.  up-right-down-left	move horizontally in x-y plane
   4.  .-;	move vertically
   5.  o-p	rotate (yaw)
   6.  y-h	rotate (pitch)
   7.  e-r	rotate (roll)
   8.  b	toggle arm/base mode (if appli cable)
   9.  s	switch active arm (if multi-armed robot)
   10. =	switch active robot (if multi-robot env)
   11. ESC	quit
10. demo_video_recording.py --environment Lift --robots Panda 录像的，需要先install imagoio,egl也要修改，参考教程里面的安装部分，根据报错还需要安装imageio[ffmpeg]或者imageio[pyav]，我也不知道哪一个好点。文件存放在执行python时的文件夹里面
11. python demo_renderers.py ，有mjviewer和mujoco两种，看不出来什么区别
12. demo_usd_export.py 过程比较繁琐，允许用户在外部渲染器（如NVIDIA Omniverse和Blender）中渲染robosuite轨迹
    1.  `pip install usd-core pillow tqdm`
    2.  安装依赖项后，可以通过从`from robosuite.utils.usd import exporter`导入USD导出器。exporter模块中的USDExporter类处理导出与robosuite轨迹关联的所有必要资产和USD文件。
    3.  尝试运行，啥都没有，有个警告If using later versions of mujoco, please use the exporter in the mujoco repository: https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/usd/exporter.py，下载下来代替原来的exporter.py（控制台的报错信息可以定位到），最后保存的文件位置也是在执行的目录里面

给gym单开一个

1. 使用gym的话要先安装，可以用示例代码
2. 运行的时候可能会报错，没安装好，可以参考[csdn，安装gym[box2d]失败](https://blog.csdn.net/tortorish/article/details/131374265)
   1. `pip install gymnasium[all]`
   2. 遇到报错ERROR: Could not build wheels for box2d-py, which is required to install pyproject.toml-based projects：`conda install swig`
   3. 遇到报错error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
      1. 从[visualstudioc++](https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools/)直接点下载生成工具
      2. 选使用C++的桌面开发，点击安装。
   4. 然后重新装一下
3. 测试代码
   ```python
   import gymnasium as gym
   env = gym.make("LunarLander-v2", render_mode="human")
   observation, info = env.reset(seed=42)
   for _ in range(1000):
      action = env.action_space.sample()  # this is where you would insert your policy
      observation, reward, terminated, truncated, info = env.step(action)
   
      if terminated or truncated:
         observation, info = env.reset()
   env.close()
   ```




# modules

## overview
2大类api

1. 以模块化和编程方式定义仿真环境的**建模api**
2. 用于与外部输入（例如来自策略或I/O设备的输入）接口的**模拟API**。

这边模拟和仿真都是simulation

由建模API指定的**仿真模型**由**MuJoCo引擎**实例化以创建称为**环境**的仿真运行时 (simulation runtime)。环境通过传感器（例如摄像头和本体感受）生成观察结果，并通过机器人的控制器接收来自策略或设备的动作命令。

![框架](https://robosuite.ai/docs/_images/module_overview.png)

里面包含Task,Environment

simulation model 由 Task object 定义，封装了机器人仿真的三个基本组成部分：

1. robot model：等于等于1个，加载机器人的模型，也可以选择加载其他模型
2. object model：可有可无，可以从3D对象资源加载，也可以通过编程API以程序方式生成。
3. arena：只能有一个，定义机器人的工作空间，包括环境固定装置（如桌面）及其放置位置。

任务类将这些成分组合成MuJoCo的MJCF建模语言中的单个XML对象。该MJCF对象通过mujoco库传递给MuJoCo引擎，以实例化MjModel对象以供模拟运行时使用。


Environment对象为外部输入提供OpenAI Gym-style API，以便与仿真进行交互。

外部输入对应于用于控制机器人及其所拥有的任何致动器的动作命令，其中动作空间的运动分量特定于机器人所使用的控制器。这些动作命令可以由算法自动生成，也可以来自用于人类远程操作的I/O设备。

机器人的控制器负责解释这些动作命令，并将其转换为传递到底层物理引擎的低级别扭矩，该引擎执行内部计算以确定仿真的下一个状态。传感器从MjSim对象中检索信息，并生成观察结果，作为机器人接收的物理信号，作为对其动作的响应。我们的框架支持多模态传感模式，并提供模块化API来模拟真实的传感器动态。除了这些感官数据，环境还提供有关任务进度和成功条件的其他信息，包括奖励函数和其他元数据。

## robots
robots对应overview图中Task内部的Robot model。Robots作为给定代理的体现以及环境中的中心交互点和MuJoCo的关键接口，用于机器人相关的状态和控制。robosuite通过基于Robot的类捕获了这种抽象级别，支持固定基座和移动基座变体。Robot类由**RobotModel、RobotBaseModel和Controller**集中定义。RobotModel类的子类还可以包括其他模型。

Rivirs能实现的功能有：

1. 多样化和现实的模型：robosuite提供了20个商用机器人（包括人形GR 1机器人），15个抓取器（包括inspire灵巧手模型）和6个控制器的模型，模型属性直接来自公司网站或原始规格表。
2. 模块化支持：机器人被设计为即插即用-机器人，模型和控制器的任何组合都可以使用，假设给定的环境旨在用于所需的机器人配置。由于每个机器人都被分配了一个唯一的ID号，因此相同机器人的多个实例可以在仿真中实例化而不会出错。
3. 自封闭抽象：对于给定的任务和环境，与特定机器人实例相关的任何信息都可以在该实例的属性和方法中找到。这意味着每个机器人负责在每个情节开始时直接在模拟中设置其初始状态，并且还通过其控制器的转换动作输出的扭矩直接控制模拟中的机器人。

### 用法
#### 初始化
在环境创建过程中（suite.make(...)），各个机器人会被实例化和初始化。所需的机器人模型（RobotModel）、坐骑模型（MountModel）和控制器（Controller）（可指定多个和/或附加模型，例如用于机械手双臂机器人）将被加载到每个机器人中，这些模型将被传递到环境中，以组成最终的 MuJoCo 仿真对象。然后将每个机器人设置为初始状态。

#### 运行时
在给定的模拟事件期间（env.step(...)调用），环境将接收一组动作，并根据它们各自的动作空间将它们相应地分配给每个机器人。然后，每个机器人通过各自的控制器将这些动作转换为低级别的扭矩，并在仿真中直接执行这些扭矩。在环境步骤结束时，每个机器人将其机器人特定的观测值集传递给环境，然后环境将连接并附加额外的任务级观测值，然后将它们作为env.step(...)的输出

#### 调用对象
在任何给定时间，每个机器人都有一组properties，可以随时访问其实时值。这些包括给定机器人的规格，例如其DoF，动作尺寸和扭矩限制，以及本体感受值，例如其关节位置和速度。此外，如果机器人启用了任何传感器，也可以轮询这些读数。机器人属性的完整列表可以在机器人API部分找到。


### 模型


# Algorithms
## Benchmark
v1.0是最新的吧？v0.3早一点。

也就是说robosuite+sac需要2天才能跑完一个实验，我应该不用跑吧……

1. v1.0
   1. robosuite提供的benchmark 
      1. 利用强化学习库 rlkit
      2. 测试 Soft Actor-Critic，无模型rl算法 （2018年）
   2. 测试结果[github:benchmarking for robosuite+SAC](https://github.com/ARISE-Initiative/robosuite-benchmark)
   3. 效果
      1. 所有代理都经过了500个epoch的训练，每个epoch有500个步骤，并使用相同的标准化算法超参数
      2. 这些实验在2个CPU和12 G VRAM上运行，没有GPU，每个实验大约需要两天才能完成。
   4. 结论：用panda(osc)
2. v0.3
   1. 查阅Surreal论文，其他啥都没说
   2. 还提供了一个思路RoboTurk数据集上的模仿学习





### 代码框架
> windows和wsl上都失败了orz，我试试阅读下代码吧，了解一下框架

之前根据教程，确认运行实验是运行scripts/train.py，其他可视化环节先不管(也可以顺便看一下为什么我windows上面的log地址能不能手动改一下，好吧一开始初始化就有问题。。)

1. train.py执行先配置参数，后面到utils/rlkit_utils.py执行experiment.py
2. rlkit.py里面有用到robosuite相关的包，*尤其是GymWrapper*
   1. tran调用的是rlkit.py里面的experiment，参数varient
      1. 对每个参数
         1. 根据参数设置controller
         2. 设置robosuite环境并加入到suites中
      2. GymWrapper 封装 suites
      3. MLP
      4. 用agent实例化tran













跑实验[github benchmark:robosuite+SAC](https://github.com/ARISE-Initiative/robosuite-benchmark)

**benchmark这个版本只能在linux或者mac上面跑，windows真不行**

我非常怀疑是因为这个benchmark自己的问题:<

-----------------

### WSL配置 fail纪念
用WSL 18.04跑的

1. 根据github read-me操作
2. 安装anaconda
3. pip install requirements时出现问题，一看果然又是mujoco，建议先用`pip install mujoco==2.0.2.9`试试
   1. command ‘gcc‘ failed with exit status 1报错
      1. gcc --version发现没有，根据它的提示安装
      2. ubuntu18.04默认安装7的gcc，看起来能用
   2. 再运行install，出现新报错fatal error: ffi.h: No such file or directory
      1. 执行`sudo apt-get install build-essential libssl-dev libffi-dev python-dev`
   3. 新报错 missing MuJoCo，太熟悉了
      1. 下载mujo200: sudo wget https://www.roboti.us/download/mujoco200_linux.zip
      2. unzip ，需要先安装unzip 到 home下的.mujoco文件夹里面
         1. ` mkdir ~/.mujoco `
         2. 并把解压后的文件夹命名为mujoco200 
      3. 下载key：sudo wget https://www.roboti.us/file/mjkey.txt
      4. 移动license
         1. `cp mjkey.txt ~/.mujoco`
         2. `cp mjkey.txt ~/.mujoco/mujoco200/bin`
      5. 测试
         1. `cd ~/.mujoco/mujoco200/bin `
         2. `./simulate ../model/humanoid.xml`
4. 继续`pip install -r requirements.txt`这下应该没问题了，之前windows上面也主要是
5. 出现了这样的报错……  ERROR: Could not find a version that satisfies the requirement mink>=0.0.5 (from robosuite>=1.0.1->-r requirements.txt (line 1)) (from versions: none)
ERROR: No matching distribution found for mink>=0.0.5 (from robosuite>=1.0.1->-r requirements.txt (line 1))，先搞定别的
6. 安装完其他的之后从github里找个配置文件的例子，跑一下
   1. 安装gtimer模块，pip install
   2. 不知道为什么找不到gym，再根据requirements安装下
      1. 报错，说要mjpro150，我服了，再干一遍
      2. 还是有问题说Failed building wheel for atari-py：sudo apt install cmake libz-dev
      3. 提示Please add following line to .bashrc:export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/bmtest/.mujoco/mjpro150/bin
      4. 报错，GL/osmesa.h文件：sudo apt install libosmesa6-dev
7. 还是报错，requirement.txt上 gym和mujoco-py的版本不一致 ;-;
8. 而且mujoco150和mujoco200到底安装哪个我也不清楚，烦


-------------

### windows配置 fail纪念
1. 根据说明clone
2. 新建环境，它没有windows的，我按照mac来，因为linux比mac就多最后一行
3. 为了方便使用也在idea上设置一下
4. 先帝创业未半而中道崩殂,老版本都比较贴近linux
5. 其他都能搞完，主要是mujoco和mujoco-py有点难搞
   1. mujoco:150
      1. [下载](https://www.roboti.us/index.html)，本体和license都要，找150的
      2. 解压之后 mjkey.txt放到 \mujoco和\mujoco\bin\里面
      3. mujoco文件夹的位置和命名需要放在c:\users\..里面，必须
      4. 检测，能出现图像才算胜利
      ```
      cd C:\Users\xxx\.mujoco\mjpro150\bin
      simulate.exe ../model/humanoid.xml
      ```
   2. 配置mujoco：[参考](https://blog.51cto.com/u_15346769/5275646)，添加系统变量后要重启
   3. mujoco-py
      1. 因为github上面找不到这个版本的，只能下载到本地后移动文件夹到虚拟环境中
      2. [参考](https://blog.51cto.com/u_15346769/5275646)
      3. import mujoco_py 一直报缺少一个头文件的问题，后面解决了是要200的mujoco，把原来150的换一下，再改环境变量，import没问题了orz，希望robosuite和gym不会出问题……看起来这下mujoco-py没问题了
   4. 试了下import gym和robosuite，暂时还没报错，继续这么下去吧。
6. 跟着github的readme搞剩下的，很简单
7. export PYTHONPATH=.:$PYTHONPATH 要改成set，export是linux的
8. google这个https://blog.csdn.net/qq_42145681/article/details/116567365
9. 运行时遇到import util错误，在train.py的from util...之前加上sys.path.append('.')，要在根目录执行，如果要在scripts目录下运行的话就要append两个点
10. UnicodeEncodeError: 'locale' codec can't encode character '\u4e2d' in position 25: encoding error，根据这个，remove %Z







# 比较新的使用这个平台的算法
1. airhockey_challenge_robosuite
   1. 又是一个不写python版本的项目，target-version = ["py36", "py37", "py38"]，那我用py38
   2. conda create -n airhockey python==3.8
   3. conda activate airhockey,pip3 install -e .
   4. 尝试运行，报错ImportError: Cannot initialize a EGL device display. This likely means that your EGL driver does not support the PLATFORM_DEVICE extension, which is required for creating a headless rendering context. 烦，egl到底是个什么鬼东西
   5. lspci | grep -i vga 输出VGA compatible controller: Intel Corporation TigerLake-LP GT2 [Iris Xe Graphics] (rev 01)
   6. 是intel的,接下来装mesa
   7. sudo apt update
   8. 是不是设备的问题没法用egl阿,不想安装egl了，麻烦，不知道为什么上午成功了。。
2. cec_main
   1. conda create -n cec python==3.8
   2. pip install -r requirements.txt
   3. 报错：Cargo, the Rust package manager, is not installed or is not on PATH.This package requires Rust and Cargo to compile extensions. Install it through the system's package manager or via https://rustup.rs/解决：curl https://sh.rustup.rs -sSf | sh，关闭终端后再打开
   4. 再pip install requirements，有点慢
   5. 后面很顺利，就是时间很长
   6. 下一阶段准备数据，看了下机器人相关的数据是robomic
   7. robomimic这个数据集根robosuite有关系阿，以后详细看看https://robomimic.github.io/docs/datasets/overview.html
   8. 怎么是个github仓库，我的数据呢，好吧先安装仓库，python环境一样就不用新建环境了
   9. 居然还没装git，我装我装sudo apt install git
   10. 顺便安装了向日葵，以后如果要卸载的话https://sunlogin.oray.com/download/linux?type=personal 向日葵还是山了把
   11. 下载件夹金错乐 ，应该是git clone https://github.com/NVlabs/mimicgen.git
   12. sudo apt install build-essential 用这个解决egl_probe安装错误的问题
   13. 后面的正常了，还好
   14. 还要安装robosuite，老熟人了
   15. 下载的时候解决下huggingface_之后install，报错cmake，pip安装
   16. 还是报错：Failed to build egl_probe ERROR: ERROR: Failed to build installable wheels for some pyproject.toml based projects (egl_probe)
   17. 再安装make sudo apt install make
   18. 实施pip install --upgrade setuptools，失败
   19. 不太好从github仓库下载，卸载了，还是直接网上下载数据集，直接huggingface，打不开
   20. woc，文hub.errors.LocalEntryNotFoundError: An error happened while trying to locate the file on the Hub and we cannot find the requested files in the local cache. Please check your connection and try again or make sure your Internet connection is on.
   21. 网络问题，用镜像解决https://blog.csdn.net/weixin_44257107/article/details/136532423，
   22. 执行的命令行不知道怎么写，https://github.com/CEC-Agent/CEC/issues/1 issue有提到，但我同样出现报错
   23. Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace. 针对这个问题，参考下这个https://blog.csdn.net/weixin_44212848/article/details/138118183，但我是ubuntu
3. roboai
   1. 看了readme，有好几个task，我要看Robot going from plain text to grasping attempt -- integrated with ROS2, MoveIt2, a grasping model
   2. robosuite mujoco 相关主要在/robosim 。没有这个文件夹。又没说到底要python几
   3. 啥都没，装个p
4. robosuite_crise137
   1. conda create -n crise python==3.8.16 pip git
   2. 没acitivae，直接安装，命令行好像没法换行
   3. 安装mjoco210
   4. conda activate crise
   5. 安装完之后去robosuite/CRiSE137，robosuite文件夹和之前看的结构一样唉
   6. 尝试运行一个，报killed。猜测是oom的问题，检查了一下，https://blog.csdn.net/Castlehe/article/details/122936585参考这个
   [26593.471405] oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null),cpuset=/,mems_allowed=0,global_oom,task_memcg=/user.slice/user-1000.slice/user@1000.service/app.slice/snap.intellij-idea-community.intellij-idea-community-5e3c6e34-4572-44b5-bd50-e3d1cd6c9a9d.scope,task=python3,pid=36300,uid=1000
               [26593.471569] Out of memory: Killed process 36300 (python3) total-vm:42993832kB, anon-rss:10639968kB, file-rss:384kB, shmem-rss:0kB, UID:1000 pgtables:23252kB oom_score_adj:0
5. robosuite_models_main
   1. 尝试运行robosuite_models-main
    1. codna:models,python=3.9
        1. 报错：no module named mujoco,pip install
        2. no module named robosuite,pip install
        3. 安装robosuite时报错 The 'linux/input.h' and 'linux/input-event-codes.h' include files
                are missing. You will have to install the kernel header files in
                order to continue:
                    dnf install kernel-headers-$(uname -r)
                    apt-get install linux-headers-$(uname -r)
                    emerge sys-kernel/linux-headers
                    pacman -S kernel-headers
        4. 运行了 apt-get install linux-headers-$(uname -r)，同样的报错
        5. 安装dnf后运行第一条指令
            出现新报错：Unable to detect release version (use '--releasever' to specify release version)
                  Error: There are no enabled repositories in "/etc/yum.repos.d", "/etc/yum/repos.d", "/etc/distro.repos.d".
                  艘了下好像是centos的，我是ubuntu阿，用apt-get remove卸载dnf
          第三个不知道是什么指令
          第四个 apt install pacma，运行后像个小游戏。。然后继续install robosuite失败
          是不是gcc的问题阿，apt install gcc，woc还真是
         6. EGL: EGL is not or could not be initialized'
              warnings.warn(message, GLFWError)
            ERROR: could not create window
          7. sudo apt -y install libnvidia-egl-wayland1 安装egl试试
          报错一样,卸载

          8. MUJOCO_GL=osmesa python examples/load_compositional_robot_example.py --robots UR5eOmron  --composite-controller HYBRID_MOBILE_BASE
          pip install --upgrade pyrender
          继续出现报错
          sudo apt install  libosmesa6  libosmesa6-dev
          继续报错
          https://blog.csdn.net/CCCDeric/article/details/129292944 根据这个，过程一样，改个用户名
          总算是有东西了，虽然参数不完全对

         问了gpt之后能安装egl了
         然后执行MUJOCO_GL=osmesa python examples/load_compositional_robot_example.py --robots UR5eOmron  --composite-controller HYBRID_MOBILE_BASE
         会出现 composite-controller 没有这个参数

         github上的示例代码怎么改了？python examples/load_compositional_robot_example.py --robots UR5eOmron  --controller BASIC 现在是这个，但我不知道为什么我的运行起来也怪怪的

conda activate models
python examples/load_compositional_robot_example.py --robots UR5eOmron  --controller BASIC

6. Robotic-Softbody-Manipulation-master
   1. 该框架的首要目标是使机器人操纵器能够进行自动超声成像。
作为第一步，已经创建了用于软接触实验的模拟环境。使用强化学习，超声任务的目标是学习机器人如何在施加所需的力并保持恒定速度的同时在柔软物体的表面上进行扫描。

可以训练RL代理来执行超声任务，其中框架已与稳定基线的PPO算法集成。


总结

1. robosuite_models_main可以弹出窗口，但是运行起来感觉不像是跑了什么程序的样子
2. cec-main 虽然我辛辛苦苦把数据下载下来了，但是运行不起来，命令行不知道怎么写
3. dexnet_robosuite=master连readme都没有，不跑，有命令行，但是不知道怎么办
4. assistive-gym-robosuite-master readme没有，requirement.txt报错，放弃了
5. roboai readme相当于没写
6. robosuite_CRiSE137 安装挺好的，就是oom，过会儿看看
7. robosuite-multi-agent-RL benchmark改的，我benchmark都没跑起来
8. airhockey-challenge-robosuite egl安装不起来。找不到原因
9. dexnet_rrt_planner_surreal_robosuite 又不懈要求
10. diffusion_policy_with_robosuite 只有一个ipynb，看pdf好像还可以，有空再测试，它这个给了个流程图，数据收集部分需要自己干
11. model-bases-rl-gazebo-robosuite 程序中很多ipynb，我不太好运行
看到现在我大概了解了robosuite在这些项目的哪个位置，接下来我想看看gym和gymwrapper出现在哪儿，怎么用的
12. reinforcement-learning-sandbox 这个就是单纯强化学习的库吧,4年前的，没有更新，跳过吧，还要用docker，很麻烦
13. robosuite-panda-ik 项目太老了
14. robosuite-tasl-zoo egl问题
15. robosuite-playground 又啥都不写还有docker，直接跳过
16. robosuite_research_project_template 生成数据的？不懂，暂时用不上
17. robotic-softbody-manipulation robosuite都没用。。有mujoco+强化学习框架
18. safe-multi-agent-robosuite 这个也需要安装robosuite，干脆重新开一个吧


robosuite重新配置

1. python==3.10，robosuite==1.5.0
2. 在work文件夹中安装robosuite：git clone https://github.com/ARISE-Initiative/robosuite.git
   1. conda create -n robosuite python=3.10 
   2. cd robosuite 
   3. conda activate robosuite
   4. pip3 install -r requirements.txt
   5. pip3 install -r requirements-extra.txt
   6. python robosuite/demos/demo_random_action.py 测试成功
2. 在demos/test1_simple.py里面实现一个简单的键盘控制，理解demo_control的代码
3. 整理下robosuite提供的各个参数的可选项
   1. Registered environments: Lift, Stack, NutAssembly, NutAssemblySingle, NutAssemblySquare, NutAssemblyRound,
      PickPlace, PickPlaceSingle, PickPlaceMilk, PickPlaceBread, PickPlaceCereal,
      PickPlaceCan, Door, Wipe, TwoArmLift, TwoArmPegInHole, TwoArmHandover
   2. Possible robots: Baxter, IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e
   3. nutassembly好像有摩擦力的问题。
   4. baxter怎么上来就是俩手臂
4. environments/manipulation/里有environment这个参数能用的几个选项，要修改环境的话就是看怎么修改这个。nut相关的5个里面比较全
   1. 为什么environment里面只有一个nut_assembly阿，另外四个呢？哦都在一个文件里。我真的很好其在哪里注册的，正好这个类名字比较特殊
      1. 物体位置 placement_initializer
      2. 物体模型 MujocoObject https://robosuite.ai/docs/modules/objects.html
      4. 相机 demo_domain_randomization，DomainRandomizationWrapper
5. 怎么重写以获得一个自己定义的类（要看别的项目有没有相关代码） https://robosuite.ai/docs/tutorials/add_environment.html
   1. 最好能在外部修改
   2. 改不了看看能不能新建一个自己的类
   3. 下下策是直接修改某个类
6. 外部操控的代码
   1. 参考gpt
   2. 不知道，搜一下
7. environment代码里面有reward和model，有没有搞头
8. controller部分看怎么控制机械臂力量






