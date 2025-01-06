---
title: genesis平台
tags:
  - robotic
mathjax: false
categories: it
date: 2024-12-19 14:51:53
---

一个比robosuite更简洁的物理平台，专为通用机器人/人工智能/物理AI应用而设计。

<!--more-->

[教程](https://genesis-world.readthedocs.io/en/latest/index.html)

# install
1. 需求：
   1. Python: 3.9+ 所以我`conda create -n genesis python==3.10`
   2. OS: Linux (recommended) / MacOS / Windows 但windows不太方便。
2. ~~`pip install genesis-world` ~~ 因为这个项目还在即时更新，建议用github clone方式安装
   1. ~/work文件夹里面
   2. ` git clone https://github.com/Genesis-Embodied-AI/Genesis.git`
   3. `export PYTHONPATH=~/work/Genesis` 这个是修改python路径的，不是永久的。
   4. 如果需要经常使用的话
      1. `sudo vim ~/.bashrc`
      2. 加入`export PYTHONPATH=~/work/Genesis:$PYTHONPATH`保存
   5.  在work里创建一个文件夹写代码。可以调用Genesis下面的genesis
3. 安装[pytorch](https://pytorch.org/get-started/locally/)
4. 以上就是最基础的安装，还有一些额外的根据需求安装即可
5. 更新：因为genesis还在更新中，偶尔需要更新
   1. ~~`python -m pip install --upgrade pip`~~
   2. ~~`pip install --upgrade 包名称`~~
   3. 在Genesis里`git pull`



# 基础使用
## 开始
```python
import genesis as gs
# 初始化
gs.init(
    seed                = None,
    precision           = '32',
    debug               = False,
    eps                 = 1e-12,
    logging_level       = 'warning',
    # logging_level       = None,
    backend             = gs.cpu,
    theme               = 'light',
    logger_verbose_time = False
)

# 场景包装一个模拟器对象（处理所有底层物理解算器）和一个可视化工具对象
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,
        gravity=(0, 0, 0.0),
    ),
    show_viewer=True,
    # camera_fov
    viewer_options = gs.options.ViewerOptions(
        res           = (1280, 960),
        camera_pos    = (3.5, 0.0, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 40,
        max_FPS       = 60,
    ),
    vis_options = gs.options.VisOptions(
        show_world_frame = True, # visualize the coordinate frame of `world` at its origin
        world_frame_size = 1.0, # length of the world frame in meter
        show_link_frame  = False, # do not visualize coordinate frames of entity links
        show_cameras     = False, # do not visualize mesh and frustum of the cameras added
        plane_reflection = True, # turn on plane reflection
        ambient_light    = (0.1, 0.1, 0.1), # ambient light setting
    ),
    renderer = gs.renderers.Rasterizer(), # using rasterizer for camera rendering
)
# load object
# 在创建变形时，您还可以指定其位置、方向、大小等。对于方向，变形接受eXtreme（scipy外部x-y-z约定）或quat（w-x-y-z约定）。
plane = scene.add_entity(
    gs.morphs.Plane()
)

franka = scene.add_entity(
    gs.morphs.MJCF(
        file  = 'xml/franka_emika_panda/panda.xml',
        pos   = (0, 0, 0),
        euler = (0, 0, 45), # we follow scipy's extrinsic x-y-z rotation convention, in degrees,
        #     # quat  = (1.0, 0.0, 0.0, 0.0), # we use w-x-y-z convention for quaternions,
        scale = 1,
    ),
)


scene.build()

# 运行多久全看这个range？没有自己退出的功能？
for i in range(100000):
    scene.step()
```

分析

1. 导入genesis并初始化
2. 创建场景Scene：一个场景包含一个simulator对象，处理所有底层物理求解器，以及一个visualizer对象，管理与可视化相关的概念。创建场景时，可以配置各种物理求解器参数。
3. 将对象加载到场景中，所有对象和机器人都表示为Entity，Genesis设计为完全面向对象：用scene.add_entity
4. 构建场景并开始模拟：scene.build() 和 step

## 可视化
设置相机参数

```python
import genesis as gs

gs.init(backend=gs.cpu)

scene = gs.Scene(
    show_viewer = True,
    viewer_options = gs.options.ViewerOptions(
        res           = (1280, 960),
        camera_pos    = (3.5, 0.0, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 40,
        max_FPS       = 60,
    ),
    vis_options = gs.options.VisOptions(
        show_world_frame = True,
        world_frame_size = 1.0,
        show_link_frame  = False,
        show_cameras     = False,
        plane_reflection = True,
        ambient_light    = (0.1, 0.1, 0.1),
    ),
    renderer=gs.renderers.Rasterizer(),
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

cam = scene.add_camera(
    res    = (640, 480),
    pos    = (3.5, 0.0, 2.5),
    lookat = (0, 0, 0.5),
    fov    = 30,
    GUI    = False,
)

scene.build()

# 渲染rgb、深度、分割掩码和法线图
# rgb, depth, segmentation, normal = cam.render(rgb=True, depth=True, segmentation=True, normal=True)

cam.start_recording()
import numpy as np

for i in range(120):
    scene.step()
    cam.set_pose(
        pos    = (3.0 * np.sin(i / 60), 3.0 * np.cos(i / 60), 2.5),
        lookat = (0, 0, 0.5),
    )
    cam.render()
cam.stop_recording(save_to_filename='video.mp4', fps=60)
```

## 控制robot
```python
import numpy as np

import genesis as gs

########################## 初始化 ##########################
gs.init(backend=gs.cpu)

########################## 创建场景 ##########################
scene = gs.Scene(
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (0, -3.5, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 30,
        res           = (960, 640),
        max_FPS       = 60,
    ),
    sim_options = gs.options.SimOptions(
        dt = 0.01,
    ),
    show_viewer = True,
)

########################## 实体 ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)
franka = scene.add_entity(
    gs.morphs.MJCF(
        file  = 'xml/franka_emika_panda/panda.xml',
    ),
)
########################## 构建 ##########################
scene.build()

jnt_names = [
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
    'joint7',
    'finger_joint1',
    'finger_joint2',
]
dofs_idx = [franka.get_joint(name).dof_idx_local for name in jnt_names]

############ 可选：设置控制增益 ############
# 设置位置增益
franka.set_dofs_kp(
    kp             = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    dofs_idx_local = dofs_idx,
)
# 设置速度增益
franka.set_dofs_kv(
    kv             = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    dofs_idx_local = dofs_idx,
)
# 设置安全的力范围
franka.set_dofs_force_range(
    lower          = np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    upper          = np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
    dofs_idx_local = dofs_idx,
)
# 硬重置
for i in range(150):
    if i < 50:
        franka.set_dofs_position(np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]), dofs_idx)
    elif i < 100:
        franka.set_dofs_position(np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]), dofs_idx)
    else:
        franka.set_dofs_position(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), dofs_idx)

    scene.step()

# PD控制
for i in range(1250):
    if i == 0:
        franka.control_dofs_position(
            np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]),
            dofs_idx,
        )
    elif i == 250:
        franka.control_dofs_position(
            np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]),
            dofs_idx,
        )
    elif i == 500:
        franka.control_dofs_position(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            dofs_idx,
        )
    elif i == 750:
        # 用速度控制第一个自由度，其余的用位置控制
        franka.control_dofs_position(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])[1:],
            dofs_idx[1:],
        )
        franka.control_dofs_velocity(
            np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0])[:1],
            dofs_idx[:1],
        )
    elif i == 1000:
        franka.control_dofs_force(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            dofs_idx,
        )
    # 这是根据给定控制命令计算的控制力
    # 如果使用力控制，它与给定的控制命令相同
    print('控制力:', franka.get_dofs_control_force(dofs_idx))

    # 这是自由度实际经历的力
    print('内部力:', franka.get_dofs_force(dofs_idx))

    scene.step()
```

1. 从第0步到第500步，我们使用位置控制来控制所有自由度，并依次将机器人移动到3个目标位置。注意，对于control_*API，一旦设置了目标值，它将被内部存储，你不需要在接下来的步骤中重复发送命令，只要你的目标保持不变。
2. 在第750步，我们展示了可以对不同的自由度进行混合控制：对于第一个自由度（自由度0），我们发送一个速度命令，而其余的仍然遵循位置控制命令。
3. 在第1000步，我们切换到扭矩（力）控制，并向所有自由度发送一个零力命令，机器人将再次因重力而掉落到地面。

## 逆运动学与运动规划
如何在Genesis中使用逆运动学（IK）和运动规划，并执行一个简单的抓取任务。 

```
python
"""
创建一个场景，加载你喜欢的机械臂和一个小立方体，构建场景，然后设置控制增益：
"""
import numpy as np
import genesis as gs

########################## 初始化 ##########################
gs.init(backend=gs.cpu,theme='light')

########################## 创建场景 ##########################
scene = gs.Scene(
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (3, -1, 1.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 30,
        max_FPS       = 60,
    ),
    sim_options = gs.options.SimOptions(
        dt = 0.01,
        substeps = 4, # 为了更稳定的抓取接触
    ),
    show_viewer = True,
)

########################## 实体 ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)
cube = scene.add_entity(
    gs.morphs.Box(
        size = (0.04, 0.04, 0.04),
        pos  = (0.65, 0.0, 0.02),
    )
)
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)
########################## 构建 ##########################
scene.build()

motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

# 设置控制增益
# 注意：以下值是为实现Franka最佳行为而调整的
# 通常，每个新机器人都会有一组不同的参数。
# 有时高质量的URDF或XML文件也会提供这些参数，并会被解析。
franka.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
)
franka.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
)
"""
接下来，让我们将机器人的末端执行器移动到预抓取姿态。这分两步完成：

    使用IK求解给定目标末端执行器姿态的关节位置

    使用运动规划器到达目标位置

Genesis中的运动规划使用OMPL库。你可以按照安装页面中的说明进行安装。

在Genesis中，IK和运动规划非常简单：每个都可以通过一个函数调用完成。
"""

# 获取末端执行器链接
end_effector = franka.get_link('hand')

# 移动到预抓取姿态
qpos = franka.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.65, 0.0, 0.25]),
    quat = np.array([0, 1, 0, 0]),
)
# 夹爪打开位置
qpos[-2:] = 0.04
path = franka.plan_path(
    qpos_goal     = qpos,
    num_waypoints = 200, # 2秒持续时间
)
# 执行规划路径
for waypoint in path:
    franka.control_dofs_position(waypoint)
    scene.step()

# 让机器人到达最后一个路径点
for i in range(100):
    scene.step()

"""
如你所见，IK求解和运动规划都是机器人实体的两个集成方法。对于IK求解，你只需告诉机器人的IK求解器哪个链接是末端执行器，并指定目标姿态。然后，你告诉运动规划器目标关节位置（qpos），它会返回一个规划和平滑的路径点列表。注意，在我们执行路径后，我们让控制器再运行100步。这是因为我们使用的是PD控制器，目标位置和当前实际位置之间会有一个差距。因此，我们让控制器多运行一段时间，以便机器人能够到达规划轨迹的最后一个路径点。

接下来，我们将机器人夹爪向下移动，抓取立方体并将其抬起：
"""
# 到达
qpos = franka.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.65, 0.0, 0.135]),
    quat = np.array([0, 1, 0, 0]),
)
franka.control_dofs_position(qpos[:-2], motors_dof)
for i in range(100):
    scene.step()

# 抓取
franka.control_dofs_position(qpos[:-2], motors_dof)
franka.control_dofs_force(np.array([-1, -1]), fingers_dof)

for i in range(100):
    scene.step()

# 抬起
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.3]),
    quat=np.array([0, 1, 0, 0]),
)
franka.control_dofs_position(qpos[:-2], motors_dof)
for i in range(200):
    scene.step()
```
## 调试
1. `pip install ipython`

有bug,终止交易.等作者改

# entity
## 物体模型
### 内部模型
#### xml(MJCF)
根据代码
```python
franka = scene.add_entity(
    gs.morphs.MJCF(
        file  = 'xml/franka_emika_panda/panda.xml',
        pos   = (0, 0, 0),
        euler = (0, 0, 45), # we follow scipy's extrinsic x-y-z rotation convention, in degrees,
        #     # quat  = (1.0, 0.0, 0.0, 0.0), # we use w-x-y-z convention for quaternions,
        scale = 1,
    ),
)
```

利用`file`参数从`xml/`导入，xml从当时安装genesis的env中找。以linux为例，这个项目的xml文件在`work/Genesis/genesis/assets/xml`里面

除了franka_emika_panda，还有另外几个，而且panda也有好几个选项。从外部导入文件没有fixed，就设置重力为0，在Scene初始化的时候

```python
    sim_options=gs.options.SimOptions(
        gravity=(0,0,0)
    )
```

1. franka_emika_panda 是MJCF格式的
   1. hand.xml 就一个夹持头
   2. panda_no_tendon.xml 没有腱，这是什么？
   3. scene.xml 和之前的panda有区别吗
2. franka_sim 
   1. ball.xml 啥都没
   2. bi-franka_panda.xml
   3. frank_panda.xml
   4. franka_panda_teleop.xml
   5. franka_panda_test_convex.xml
3. universal_robots_ur5e
   1. ur5e.xml 没有hand。。
4. 其它
   1. ant.xml 也是什么都没有
   2. ant_grasp_ball.xml 有一个球
   3. ant_grasp_ground.xml 也没东西
   4. one_tet.xml 无
   5. thin_box.xml 无
   6. three_joint_link.xml 有个东西
   7. two_box.xml 还是没东西
   8. two_skeleton.xml 我去，什么东西跳过去了

#### urdf
assets文件夹里面还有其它形式的。都尝试尝试

尝试了下示例代码，有点困难

### 外部导入
1. 文件格式 xml
2. 可以从形状原语shape primitives, meshes, URDF, MJCF, Terrain, or soft robot description files、网格、URDF、MJCF、地形或软体机器人描述文件中实例化Genesis实体。
3. 待续，我尝试从mujoco提供的模型示例导入，失败
4. 加入`contype="0" conaffinity="0"` 在xml 的\<geom\>中 可能有用吧

# 操作
操作流程不复杂，给pos和quat让它自计算ik就行。但问题是pos和quat可能会越界，不知道这个怎么解决。我本来想用gs view给的上下限倒推pos的，但是很奇怪它给的get_pos的返回的都不是我想要的值。

## 控制的基本流程
1. 场景搭建
   1. 地面
   2. 机械臂
   3. 目标物体
2. 定义机械臂的关节索引和控制器参数（限制）
3. ik求解关节角度、规划运动路径、执行，最好在这里同时打开hand，就是设置qpos[:-2]最后两位的值
   ```python
   qpos = franka.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.65, 0.0, 0.25]),
    quat = np.array([0, 1, 0, 0]),
    )

    # 规划运动路径
    path = franka.plan_path(
        qpos_goal     = qpos,
        num_waypoints = 200, # 2秒时长
    )

    # 执行规划路径
    for waypoint in path:
        franka.control_dofs_position(waypoint)
        scene.step()

    # 等待到达最后一个路径点
    for i in range(100):
        scene.step()
   ```
4. ik求解向下移动到位置
5. 抓取

## 怎么链接到对应关节
总结一下：要翻xml文件。

### franka_emika_panda的panda
joint(关节)和dof(自由度)是两个相关概念。

以Franka机械臂为例,它的手臂有7个旋转关节,夹爪有2个平移关节,每个关节有1个自由度,总共9个自由度。更一般地,像自由关节(6自由度)或球形关节(3自由度)这样的关节有多个自由度。


### franka_sim/bi-franka_panda
1. 不知道它几个关节几个自由度：genesis内部的可以用`gs view 'xml/franka_sim/bi-franka_panda.xml'`查看
2. 现在知道它有7\*2个panda joint dof 和2\*2个finger joint dof，一共18个，然后panda joint部分是第1、3、5、7、9、11、13对应单臂，另外一半对应双臂。15、16对应单臂的夹爪。
3. 好了现在开始改ikmp部分的代码。
   1. 可视化，加载双臂机器人、物体、平面
   2. 移动机械臂关节到物体上方
   3. 向下移动机械臂
   4. 抓取

## 让robot保持静止
1. 获取各joint的初始状态
2. set joint状态

```python
staticqpos=franka.get_qpos()

for i in range(1200):
    franka.set_qpos(staticqpos)
    scene.step()
```

# 录像
[linux安装播放器](https://blog.csdn.net/qq_41251963/article/details/109676345)