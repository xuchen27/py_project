import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# 初始化参数
camera_matrix = np.array([[1000, 0, 320],
                          [0, 1000, 240],
                          [0, 0, 1]], dtype=np.float32)

rvec = np.zeros(3, dtype=np.float32)
tvec = np.zeros(3, dtype=np.float32)
point = [0, 0, 5]


def get_jacobian_column_labels(num_cols):
    """
    Generate labels for the columns of the Jacobian matrix.

    Args:
        num_cols: Number of columns in the Jacobian matrix.

    Returns:
        A list of labels for the columns.
    """
    labels = []

    # 3D points coordinates
    # if num_cols >= 3:
    #     labels.extend(['∂u/∂X', '∂u/∂Y', '∂u/∂Z'])

    # Rotation vector
    if num_cols >= 3:
        labels.extend(['∂u/∂rx', '∂u/∂ry', '∂u/∂rz'])

    # Translation vector
    if num_cols >= 6:
        labels.extend(['∂u/∂tx', '∂u/∂ty', '∂u/∂tz'])

    # Camera intrinsic parameters
    if num_cols >= 10:
        labels.extend(['∂u/∂fx', '∂u/∂fy', '∂u/∂cx', '∂u/∂cy'])

    # Distortion parameters
    if num_cols >= 14:
        labels.extend(['∂u/∂k1', '∂u/∂k2', '∂u/∂p1', '∂u/∂p2'])

    # Add more if there are distortion coefficients and other parameters
    return labels


def update_jacobian(rvec, tvec, point, camera_matrix):
    objectPoints = np.array([[point[0], point[1], point[2]]], dtype=np.float32)
    distCoeffs = np.zeros((4, 1), dtype=np.float32)
    _, jacobian = cv2.projectPoints(objectPoints, rvec, tvec, camera_matrix, distCoeffs)
    return jacobian


# 重置按钮的回调函数
def reset(event):
    slider_rvec_x.reset()
    slider_rvec_y.reset()
    slider_rvec_z.reset()
    slider_tvec_x.reset()
    slider_tvec_y.reset()
    slider_tvec_z.reset()
    slider_point_x.reset()
    slider_point_y.reset()
    slider_point_z.reset()


# 初始化图形
fig, ax = plt.subplots(figsize=(15, 10))
plt.subplots_adjust(left=0.25, bottom=0.6)
jacobian_img = ax.imshow(np.zeros((2, 14)), cmap='hot', interpolation='nearest')
plt.colorbar(jacobian_img, ax=ax)
ax.set_title('Jacobian Matrix Visualization')
ax.set_xlabel('Parameters')
ax.set_ylabel('2D Point Components')

# 添加滑动条
ax_rvec_x = plt.axes([0.25, 0.50, 0.65, 0.03])
ax_rvec_y = plt.axes([0.25, 0.45, 0.65, 0.03])
ax_rvec_z = plt.axes([0.25, 0.40, 0.65, 0.03])
ax_tvec_x = plt.axes([0.25, 0.35, 0.65, 0.03])
ax_tvec_y = plt.axes([0.25, 0.30, 0.65, 0.03])
ax_tvec_z = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_point_x = plt.axes([0.25, 0.20, 0.65, 0.03])
ax_point_y = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_point_z = plt.axes([0.25, 0.10, 0.65, 0.03])
resetax = plt.axes([0.025, 0.025, 0.1, 0.04])

slider_rvec_x = Slider(ax_rvec_x, 'rvec_x', -3.14, 3.14, valinit=0)
slider_rvec_y = Slider(ax_rvec_y, 'rvec_y', -3.14, 3.14, valinit=0)
slider_rvec_z = Slider(ax_rvec_z, 'rvec_z', -3.14, 3.14, valinit=0)
slider_tvec_x = Slider(ax_tvec_x, 'tvec_x', -10, 10, valinit=0)
slider_tvec_y = Slider(ax_tvec_y, 'tvec_y', -10, 10, valinit=0)
slider_tvec_z = Slider(ax_tvec_z, 'tvec_z', -10, 10, valinit=0)
slider_point_x = Slider(ax_point_x, 'point_x', -5, 5, valinit=0)
slider_point_y = Slider(ax_point_y, 'point_y', -5, 5, valinit=0)
slider_point_z = Slider(ax_point_z, 'point_z', 1, 10, valinit=5)
button = Button(resetax, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')

button.on_clicked(reset)

# 初始化文本注释列表
texts = []


def update(val):
    global texts
    rvec = np.array([slider_rvec_x.val, slider_rvec_y.val, slider_rvec_z.val], dtype=np.float32).reshape(3, 1)
    tvec = np.array([slider_tvec_x.val, slider_tvec_y.val, slider_tvec_z.val], dtype=np.float32).reshape(3, 1)
    point = [slider_point_x.val, slider_point_y.val, slider_point_z.val]

    jacobian = update_jacobian(rvec, tvec, point, camera_matrix)
    jacobian_img.set_data(jacobian)

    # 清除之前的文本注释
    for text in texts:
        text.remove()
    texts = []
    num_cols = jacobian.shape[1]
    column_labels = get_jacobian_column_labels(num_cols)

    if len(column_labels) == num_cols:
        ax.set_xticks(ticks=np.arange(num_cols), labels=column_labels, rotation=0)
    ax.set_yticks(range(jacobian.shape[0]))

    for i in range(jacobian.shape[0]):
        for j in range(jacobian.shape[1]):
            text = ax.text(j, i, f"{jacobian[i, j]:.2f}", ha='center', va='center', fontsize=8,
                           color='white' if jacobian[i, j] < 0.5 else 'black')
            texts.append(text)

    plt.draw()


# 连接滑动条事件
slider_rvec_x.on_changed(update)
slider_rvec_y.on_changed(update)
slider_rvec_z.on_changed(update)
slider_tvec_x.on_changed(update)
slider_tvec_y.on_changed(update)
slider_tvec_z.on_changed(update)
slider_point_x.on_changed(update)
slider_point_y.on_changed(update)
slider_point_z.on_changed(update)

# 初始化显示
update(None)
plt.show()
