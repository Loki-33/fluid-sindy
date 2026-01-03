import pysindy as ps 
import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path 
from pydmd import DMD 
from scipy.interpolate import griddata 

class CustomLib(ps.CustomLibrary):
    def __init__(self):
        functions = [
            lambda x, y: x * y,
            lambda x, z: x * z,
            lambda y, z: y * z,
        ]
        names = [
            lambda x, y: f"{x} {y}",
            lambda x, z: f"{x} {z}",
            lambda y, z: f"{y} {z}",
        ]
        super().__init__(library_functions=functions, function_names=names)

def load_data(base_path, time_dirs, field='U'):
    snapshots = []
    coords = None 
    for t_dir in time_dirs:
        file_path = Path(base_path)/t_dir/f"{field}_wakeSlice.raw"

        if not file_path.exists():
            print(f'Warining: {file_path} not found...')
            continue 
        data = np.loadtxt(file_path)
        if coords is None:
            coords = data[:, :3]

        if field == 'U':
            field_data = data[:, 3:5].flatten()
        elif field=='p':
            field_data = data[:, 3]

        snapshots.append(field_data)

    snapshots = np.column_stack(snapshots)
    print("FILE LOADED!\n")
    print("DATA EXTRACTED\n")
    return snapshots, coords 


def perform_dmd(snapshots, dt=0.1, n_modes=10):
    print('============DMD ANALYSIS==============')
    dmd = DMD(svd_rank=n_modes)
    dmd.fit(snapshots)

    eigenvalues = dmd.eigs
    frequencies = np.imag(np.log(eigenvalues))/(2*np.pi*dt)
    growth_rates = np.real(np.log(eigenvalues))/dt 

    print(f"\nExtracted {len(frequencies)} DMD modes")
    print("\nMode Analysis:")
    print("-" * 60)
    print(f"{'Mode':<8} {'Frequency (Hz)':<18} {'Period (s)':<15} {'Growth Rate':<12}")
    print("-" * 60)
    
    for i, (freq, growth) in enumerate(zip(frequencies, growth_rates)):
        period = 1/abs(freq) if abs(freq) > 1e-6 else np.inf
        print(f"{i:<8} {freq:>15.6f}   {period:>12.3f}   {growth:>12.6f}")
    
    # Find dominant oscillatory mode (highest frequency, excluding mean mode)
    non_zero_freqs = [(i, abs(f)) for i, f in enumerate(frequencies) if abs(f) > 1e-3]
    if non_zero_freqs:
        dominant_idx = max(non_zero_freqs, key=lambda x: x[1])[0]
        dominant_freq = frequencies[dominant_idx]
        
        # Calculate Strouhal number: St = f*D/U
        # For our case: D=1, U=1
        strouhal = abs(dominant_freq) * 1.0 / 1.0
        
        print("\n" + "="*60)
        print(f"DOMINANT VORTEX SHEDDING FREQUENCY: {abs(dominant_freq):.4f} Hz")
        print(f"PERIOD: {1/abs(dominant_freq):.3f} seconds")
        print(f"STROUHAL NUMBER: {strouhal:.4f}")
        print(f"(Literature value for Re=100: St ≈ 0.16-0.17)")
        print("="*60)
    
    return dmd

def vis_dmd(dmd, coords, n_modes=4):
    print('\nGENERATING DMD MODES VIS....')
    modes = dmd.modes 
    x = coords[:, 0]
    y = coords[:, 1]

    fig, axes = plt.subplots(2, n_modes//2, figsize=(16,8))
    axes = axes.flatten()

    for i in range(min(n_modes, modes.shape[1])):
        mode_u = np.real(modes[::2, i])
        sc = axes[i].scatter(x, y, c=mode_u, cmap='RdBu_r', s=1)
        axes[i].set_title(f"DMD MODE {i}")
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')

        plt.colorbar(sc, ax=axes[i])
    plt.tight_layout()
    plt.savefig('dmd_modes.png', dpi=150, bbox_inches='tight')
    print('SAVED: dmd_modes.png')

def vis_dmd_spectrum(dmd, dt=0.1):
    eigenvalues = dmd.eigs
    frequencies = np.imag(np.log(eigenvalues)) / (2 * np.pi * dt)
    amplitudes = np.abs(dmd.amplitudes)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot eigenvalues on unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit circle')
    ax1.scatter(np.real(eigenvalues), np.imag(eigenvalues), 
                c=amplitudes, cmap='viridis', s=100, edgecolors='black')
    ax1.set_xlabel('Real')
    ax1.set_ylabel('Imaginary')
    ax1.set_title('DMD Eigenvalues')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.legend()
    
    # Plot frequency spectrum
    ax2.stem(frequencies, amplitudes, basefmt=' ')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('DMD Frequency Spectrum')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dmd_spectrum.png', dpi=150, bbox_inches='tight')
    print("Saved: dmd_spectrum.png")

def perform_sindy(snapshots, dt=0.1, coords=None):
    print('===========PERFORMING SINDy ANALYSIS==============')
    from sklearn.decomposition import PCA 

    
    snapshots = snapshots - snapshots.mean(axis=1, keepdims=True)

    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(snapshots.T).T

    print(f"Reduced to {reduced_data.shape[0]} modes")
    print(f"Explained Variance: {pca.explained_variance_ratio_.sum():.2%}")

    X = reduced_data.T 
    at = np.linspace(0, dt*(X.shape[0]-1), X.shape[0])
    library = ps.GeneralizedLibrary(
        [
            ps.IdentityLibrary(),
            CustomLib()
        ]
    )
    model = ps.SINDy(
        #feature_library=ps.PolynomialLibrary(degree=3, include_bias=False),
        feature_library=library, 
        optimizer=ps.STLSQ(threshold=1e-3),
        differentiation_method=ps.SINDyDerivative(
            kind='savitzky_golay',
            left=5*dt, 
            right=5*dt, 
            order=3
        )
    )

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X-X_mean)/X_std
    model.fit(X, t=at, feature_names=['a1', 'a2', 'a3'])

    print('DISCOVERED EQUATIONS:')
    model.print()

    print('\nSimulating')
    X_sim = model.simulate(X[0], t=at[:100])

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    for i in range(3):
        axes[i].plot(X[:, i], 'b-', label='True', linewidth=2)
        axes[i].plot(X_sim[:, i], 'r--', label='SINDy', linewidth=2)
        axes[i].set_ylabel(f'a{i+1}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time step')
    plt.suptitle('SINDy Model: True vs Predicted')
    plt.tight_layout()
    plt.savefig('sindy_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: sindy_comparison.png")
    
    return model

if __name__ == '__main__':
    BASE_PATH = 'Cylinder/postProcessing/sampleDict'
    time_dirs = [str(i) for i in range(0, 201, 10)]

dt=10.0

print('\nLoading velocity Data....')
U_snapshots, coords = load_data(BASE_PATH, time_dirs)

dmd = perform_dmd(U_snapshots, dt=dt, n_modes=10)
vis_dmd(dmd, coords, n_modes=4)
vis_dmd_spectrum(dmd, dt=dt)
sindy_model = perform_sindy(U_snapshots, dt=dt, coords=coords)

print('DONE!!!!!!!!!!!!!')
