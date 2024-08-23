import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def solve_2d_convection(u0, c_x, c_y, dx, dy, dt, nt):
    ny, nx = u0.shape
    u = u0.copy()
    u_list = [u.copy()]  # 用于存储每个时间步的结果

    for n in range(nt):
        u_new = u.copy()
        for i in range(1, nx):
            for j in range(1, ny):
                u_new[j, i] = u[j, i] - c_x * (dt / dx) * (u[j, i] - u[j, i - 1]) - c_y * (dt / dy) * (u[j, i] - u[j - 1, i])
        u = u_new
        u_list.append(u.copy())  # 保存当前时间步的结果

    return u_list

def animate_2d_convection(u_list, nx, ny):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)
    X, Y = np.meshgrid(x, y)

    def update_plot(frame):
        ax.clear()
        ax.plot_surface(X, Y, u_list[frame], cmap='viridis')
        ax.set_zlim(1, 2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('U')
        ax.set_title(f'Time step: {frame}')

    ani = animation.FuncAnimation(fig, update_plot, frames=len(u_list), interval=100)
    ani.save('2d_convection.gif', writer='imagemagick')
    plt.show()

def main():
    # 参数设置
    nx = 41  # x方向上的网格点数
    ny = 41  # y方向上的网格点数
    nt = 50  # 时间步数
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    dt = 0.01
    c_x = 1  # x方向上的对流速度
    c_y = 1  # y方向上的对流速度

    # 初始条件
    u0 = np.ones((ny, nx))
    u0[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2  # 设置初始波形

    # 求解
    u_list = solve_2d_convection(u0, c_x, c_y, dx, dy, dt, nt)

    # 绘制并保存动画
    animate_2d_convection(u_list, nx, ny)

if __name__ == "__main__":
    main()
