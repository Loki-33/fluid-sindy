from pathlib import Path 
import numpy as np 
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt 

def load_data(base_path, time_dirs, field='U'):
    snapshots = []
    times = []
    coords = None 
    
    for t_dir in time_dirs:
        file_path = Path(base_path)/t_dir/f"{field}_wakeSlice.raw"
        if not file_path.exists():
            # print(f'Warning: {file_path} not found...')
            continue 
        
        data = np.loadtxt(file_path)
        
        if coords is None:
            coords = data[:, :2]  # x, y only
        
        if field == 'U':
            field_data = data[:, 3:5]  # Keep as (N, 2)
        elif field == 'p':
            field_data = data[:, 3]
            
        snapshots.append(field_data)
        times.append(float(t_dir))  
    
    snapshots = np.array(snapshots)  # Shape: (N_times, N_points, 2)
    times = np.array(times)
    
    return snapshots, coords, times

def build_interpolators(coords, snapshots,times):
    interp_Ux = []
    interp_Uy = []
    for i in range(len(times)):
        interp_Ux.append(LinearNDInterpolator(coords, snapshots[i, :, 0]))
        interp_Uy.append(LinearNDInterpolator(coords, snapshots[i, :, 1]))
    return interp_Ux, interp_Uy

def get_velocity(x, y, t, times, interp_Ux, interp_Uy):
    if t<=times[0]:
        t = times[0]
        idx=0
        Ux = interp_Ux[0](x, y)
        Uy = interp_Uy[0](x, y)
        return np.array([Ux, Uy])
    if t >= times[-1]:
        t = times[-1]
        idx = len(times)-1 
        Ux = interp_Ux[idx](x, y)
        Uy = interp_Uy[idx](x, y)
        return np.array([Ux, Uy])

    idx = np.searchsorted(times, t) - 1

    Ux1 = interp_Ux[idx](x, y)
    Uy1 = interp_Uy[idx](x,y)

    Ux2 =interp_Ux[idx+1](x, y)
    Uy2 = interp_Uy[idx+1](x, y)
    
    w = (t - times[idx]) / (times[idx+1] - times[idx])
    Ux = (1-w)* Ux1 + w * Ux2 
    Uy = (1-w)*Uy1 + w*Uy2 

    return np.array([Ux, Uy])

def rk4(x, y, t, dt, times, interp_Ux, interp_Uy):
    v1 = get_velocity(x, y, t, times, interp_Ux, interp_Uy)

    v2 = get_velocity(x+0.5*dt*v1[0],
                      y+0.5*dt*v1[1],
                      t+0.5*dt, times, interp_Ux, interp_Uy)
    v3 = get_velocity(x+0.5*dt*v2[0],
                      y+0.5*dt*v2[1],
                      t+0.5*dt, times, interp_Ux, interp_Uy)

    v4 = get_velocity(x + dt*v3[0],
                      y+dt*v3[1],
                      t+dt,
                      times, interp_Ux, interp_Uy)

    x_new = x+(dt/6.0) * (v1[0]+2*v2[0]+2*v3[0]+v4[0])
    y_new = y+(dt/6.0)*(v1[1]+2*v2[1]+ 2*v3[1] + v4[1])
    return x_new, y_new 

def integrate_particles(x0, y0, t0, T, dt, times, interp_Ux, interp_Uy):
    x, y = x0, y0
    t = t0 
    n_steps = int(abs(T/dt))
    
    dt_step = dt if T>0 else -dt 

    for step in range(n_steps):
        v = get_velocity(x, y, t, times, interp_Ux, interp_Uy)
        if np.isnan(v[0]) or np.isnan(v[1]):
            return np.nan, np.nan
        x, y = rk4(x, y, t, dt_step, times, interp_Ux, interp_Uy)
        t += dt_step
    return x, y 

def compute_flow_map_gradient(x0, y0, t0, T, dt, epsilon, times, interp_Ux, interp_Uy):
    x_center, y_center = integrate_particles(x0, y0, t0, T, dt, times, interp_Ux, interp_Uy)
    if np.isnan(x_center):
        return None 

    x_px, y_px = integrate_particles(x0 + epsilon, y0, t0, T, dt, times, interp_Ux, interp_Uy)
    x_mx, y_mx = integrate_particles(x0 - epsilon, y0, t0, T, dt, times, interp_Ux, interp_Uy)
    x_py, y_py = integrate_particles(x0, y0 + epsilon, t0, T, dt, times, interp_Ux, interp_Uy)
    x_my, y_my = integrate_particles(x0, y0 - epsilon, t0, T, dt, times, interp_Ux, interp_Uy)    

    if any(np.isnan([x_px, x_mx, x_py, x_my])):
        return None 

    # F[0,0] = dx_final/dx_initial
    F_11 = (x_px - x_mx) / (2 * epsilon)
    # F[0,1] = dx_final/dy_initial
    F_12 = (x_py - x_my) / (2 * epsilon)
    # F[1,0] = dy_final/dx_initial
    F_21 = (y_px - y_mx) / (2 * epsilon)
    # F[1,1] = dy_final/dy_initial
    F_22 = (y_py - y_my) / (2 * epsilon)
    
    F = np.array([[F_11, F_12],
                  [F_21, F_22]])

    return F 

def compute_ftle(F, T):
    if F is None:
        return np.nan 
    # cauchy-Green strain tensor 
    C = F.T @ F 
    eigenvalues = np.linalg.eigvals(C)
    lambda_max = np.max(eigenvalues)

    ftle = (1.0/abs(T)) * np.log(np.sqrt(lambda_max))
    return ftle 

def compute_ftle_field(x_min, x_max, y_min, y_max, nx, ny, t0, T, dt, epsilon, 
                       times, interp_Ux, interp_Uy):
    x_grid = np.linspace(x_min, x_max, nx)
    y_grid = np.linspace(y_min, y_max, ny)

    X,Y = np.meshgrid(x_grid, y_grid)

    ftle_field = np.zeros((ny, nx))

    total_points = nx*ny 

    for i in range(ny):
        for j in range(nx):
            x0 = X[i, j]
            y0 = Y[i, j]

            F = compute_flow_map_gradient(x0, y0, t0, T, dt, epsilon, 
                                          times, interp_Ux, interp_Uy)

            ftle_field[i, j] = compute_ftle(F, T)

            if (i*nx+j)%100 ==0:
                print(f"Progress: {(i*nx+j)/total_points*100:.1f}")
    return X, Y, ftle_field

def plot_ftle(X, Y, ftle_field, fig_name='ftle_forward'):
    plt.figure(figsize=(12, 6))
    ftle_masked = np.ma.masked_where((np.isnan(ftle_field)) | (ftle_field < 0), ftle_field)
    vmin, vmax = 0, np.nanpercentile(ftle_field, 98)
    plt.contourf(X, Y, ftle_masked, levels=50, cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar(label='FTLE')
    
    circle = plt.Circle((0,0), 0.5, color='white', ec='black', linewidth=2, zorder=10)
    plt.gca().add_patch(circle)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Lagrangian Coherent Structures (FTLE Field)')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'{fig_name}.png', dpi=150)
    plt.close()

def plot_combined(X, Y, ftle_forward, ftle_backward):
    fig, ax = plt.subplots(figsize=(14, 6))

    ftle_masked_fwd = np.ma.masked_where((np.isnan(ftle_forward)) | (ftle_forward < 0), ftle_forward)
    ax.contour(X, Y, ftle_masked_fwd, levels=10, colors='red', linewidths=1.5, alpha=0.8)

    ftle_masked_bwd = np.ma.masked_where((np.isnan(ftle_backward)) | (ftle_backward < 0), ftle_backward)
    ax.contour(X, Y, ftle_masked_bwd, levels=10, colors='blue', linewidths=1.5, alpha=0.8)

    circle = plt.Circle((0, 0), 0.5, color='gray', ec='black', linewidth=2, zorder=10)
    ax.add_patch(circle)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('LCS: Repelling (red) and Attracting (blue)')
    ax.axis('equal')
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='red', lw=2, label='Forward-time (repelling)'),
                      Line2D([0], [0], color='blue', lw=2, label='Backward-time (attracting)')]
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig('lcs_combined.png', dpi=150)
    plt.close()

if __name__ == '__main__':
    BASE_PATH = 'Cylinder/postProcessing/sampleDict'
    time_dirs = [f"{i:.1f}" for i in np.arange(0, 20.1, 0.1)]

    snapshots, coords, times = load_data(BASE_PATH, time_dirs, field='U')
    print(f"Velocity data domain:")
    print(f"  X: {coords[:, 0].min():.2f} to {coords[:, 0].max():.2f}")
    print(f"  Y: {coords[:, 1].min():.2f} to {coords[:, 1].max():.2f}")
    print(f"  Time: {times.min():.2f} to {times.max():.2f}")
    print(f"  Loaded {len(times)} time snapshots")
    interp_Ux, interp_Uy = build_interpolators(coords, snapshots, times)

    x_min = -2.0
    x_max = 10.0 
    y_min = -3.0
    y_max = 3.0
    nx, ny = 100, 50
    t0=2.0
    T=2.0
    dt=0.005
    epsilon = 1e-6
    # FORWARD 
    X, Y, ftle_forward = compute_ftle_field(x_min,x_max,y_min,y_max,nx,ny,
                                          t0,T,dt,epsilon,times,interp_Ux,interp_Uy)
    plot_ftle(X, Y, ftle_forward)
    # BACKWARD 
    T=-2.0
    X, Y, ftle_backward = compute_ftle_field(x_min,x_max,y_min,y_max,nx,ny,
                                          t0,T,dt,epsilon,times,interp_Ux,interp_Uy)
    plot_ftle(X, Y, ftle_backward, fig_name='ftle_backward')

    #COMBINED 
    plot_combined(X, Y, ftle_forward, ftle_backward)


