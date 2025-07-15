import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import argparse
import json
import time

class F1AirflowSimulator:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.grid_size = 50
        self.dt = 0.1
        
        # Parameter simulasi
        self.wind_speed = 50.0  # m/s
        self.wind_angle = 0.0   # derajat
        self.reynolds_number = 1e6
        self.air_density = 1.225  # kg/m³
        
        # Grid untuk simulasi CFD
        self.x = np.linspace(0, 10, self.grid_size)
        self.y = np.linspace(0, 6, self.grid_size)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Velocity field
        self.u = np.zeros((self.grid_size, self.grid_size))
        self.v = np.zeros((self.grid_size, self.grid_size))
        self.pressure = np.zeros((self.grid_size, self.grid_size))
        
        # Partikel untuk visualisasi
        self.particles = []
        self.max_particles = 200
        
        # Geometri F1 (simplified)
        self.f1_geometry = self.create_f1_geometry()
        
        # Setup plot
        self.setup_plot()
        
    def create_f1_geometry(self):
        """Membuat geometri F1 yang disederhanakan"""
        geometry = {
            'body': {
                'x': [2, 8, 8, 2],
                'y': [2.5, 2.5, 3.5, 3.5]
            },
            'front_wing': {
                'x': [1.5, 2.5, 2.5, 1.5],
                'y': [2.2, 2.2, 2.4, 2.4]
            },
            'rear_wing': {
                'x': [7.5, 8.5, 8.5, 7.5],
                'y': [3.6, 3.6, 4.2, 4.2]
            },
            'wheels': [
                {'center': (2.5, 2.0), 'radius': 0.3},
                {'center': (2.5, 4.0), 'radius': 0.3},
                {'center': (7.5, 2.0), 'radius': 0.3},
                {'center': (7.5, 4.0), 'radius': 0.3}
            ]
        }
        return geometry
    
    def setup_plot(self):
        """Setup matplotlib untuk visualisasi"""
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot utama untuk streamlines
        self.ax1.set_xlim(0, 10)
        self.ax1.set_ylim(0, 6)
        self.ax1.set_aspect('equal')
        self.ax1.set_title('F1 Airflow Simulation - Streamlines', fontsize=14, fontweight='bold')
        self.ax1.set_xlabel('Posisi X (m)')
        self.ax1.set_ylabel('Posisi Y (m)')
        
        # Plot untuk pressure field
        self.ax2.set_xlim(0, 10)
        self.ax2.set_ylim(0, 6)
        self.ax2.set_aspect('equal')
        self.ax2.set_title('Pressure Field', fontsize=14, fontweight='bold')
        self.ax2.set_xlabel('Posisi X (m)')
        self.ax2.set_ylabel('Posisi Y (m)')
        
        # Gambar geometri F1
        self.draw_f1_geometry()
        
    def draw_f1_geometry(self):
        """Menggambar geometri F1 di plot"""
        # Body
        body_poly = patches.Polygon(
            list(zip(self.f1_geometry['body']['x'], self.f1_geometry['body']['y'])),
            closed=True, facecolor='red', edgecolor='black', alpha=0.8
        )
        self.ax1.add_patch(body_poly)
        
        body_poly2 = patches.Polygon(
            list(zip(self.f1_geometry['body']['x'], self.f1_geometry['body']['y'])),
            closed=True, facecolor='red', edgecolor='black', alpha=0.8
        )
        self.ax2.add_patch(body_poly2)
        
        # Front wing
        front_wing_poly = patches.Polygon(
            list(zip(self.f1_geometry['front_wing']['x'], self.f1_geometry['front_wing']['y'])),
            closed=True, facecolor='blue', edgecolor='black', alpha=0.8
        )
        self.ax1.add_patch(front_wing_poly)
        
        front_wing_poly2 = patches.Polygon(
            list(zip(self.f1_geometry['front_wing']['x'], self.f1_geometry['front_wing']['y'])),
            closed=True, facecolor='blue', edgecolor='black', alpha=0.8
        )
        self.ax2.add_patch(front_wing_poly2)
        
        # Rear wing
        rear_wing_poly = patches.Polygon(
            list(zip(self.f1_geometry['rear_wing']['x'], self.f1_geometry['rear_wing']['y'])),
            closed=True, facecolor='green', edgecolor='black', alpha=0.8
        )
        self.ax1.add_patch(rear_wing_poly)
        
        rear_wing_poly2 = patches.Polygon(
            list(zip(self.f1_geometry['rear_wing']['x'], self.f1_geometry['rear_wing']['y'])),
            closed=True, facecolor='green', edgecolor='black', alpha=0.8
        )
        self.ax2.add_patch(rear_wing_poly2)
        
        # Wheels
        for wheel in self.f1_geometry['wheels']:
            wheel_circle = patches.Circle(
                wheel['center'], wheel['radius'], 
                facecolor='black', edgecolor='gray', alpha=0.9
            )
            self.ax1.add_patch(wheel_circle)
            
            wheel_circle2 = patches.Circle(
                wheel['center'], wheel['radius'], 
                facecolor='black', edgecolor='gray', alpha=0.9
            )
            self.ax2.add_patch(wheel_circle2)
    
    def calculate_velocity_field(self):
        """Menghitung velocity field menggunakan potential flow"""
        # Base flow
        u_base = self.wind_speed * np.cos(np.radians(self.wind_angle))
        v_base = self.wind_speed * np.sin(np.radians(self.wind_angle))
        
        self.u.fill(u_base)
        self.v.fill(v_base)
        
        # Efek dari geometri F1 (simplified)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x_pos = self.X[i, j]
                y_pos = self.Y[i, j]
                
                # Efek dari body
                if 2 <= x_pos <= 8 and 2.5 <= y_pos <= 3.5:
                    self.u[i, j] = 0
                    self.v[i, j] = 0
                
                # Efek dari front wing (downforce)
                if 1.5 <= x_pos <= 2.5 and 2.2 <= y_pos <= 2.4:
                    self.u[i, j] *= 0.5
                    self.v[i, j] -= 20
                
                # Efek dari rear wing (downforce)
                if 7.5 <= x_pos <= 8.5 and 3.6 <= y_pos <= 4.2:
                    self.u[i, j] *= 0.3
                    self.v[i, j] -= 30
                
                # Efek dari wheels
                for wheel in self.f1_geometry['wheels']:
                    wheel_x, wheel_y = wheel['center']
                    wheel_r = wheel['radius']
                    distance = np.sqrt((x_pos - wheel_x)**2 + (y_pos - wheel_y)**2)
                    
                    if distance <= wheel_r:
                        self.u[i, j] = 0
                        self.v[i, j] = 0
                    elif distance <= wheel_r * 2:
                        # Turbulence effect
                        factor = 1 - (distance - wheel_r) / wheel_r
                        self.u[i, j] *= (1 - factor * 0.5)
                        self.v[i, j] += factor * 10 * np.sin(x_pos * 5)
    
    def calculate_pressure_field(self):
        """Menghitung pressure field menggunakan Bernoulli's equation"""
        velocity_magnitude = np.sqrt(self.u**2 + self.v**2)
        # Bernoulli's equation: P = P0 - 0.5 * rho * v^2
        self.pressure = 101325 - 0.5 * self.air_density * velocity_magnitude**2
    
    def update_particles(self):
        """Update posisi partikel untuk visualisasi"""
        # Hapus partikel yang sudah keluar dari domain
        self.particles = [p for p in self.particles if 0 <= p['x'] <= 10 and 0 <= p['y'] <= 6]
        
        # Tambah partikel baru
        while len(self.particles) < self.max_particles:
            self.particles.append({
                'x': np.random.uniform(0, 1),
                'y': np.random.uniform(1, 5),
                'age': 0
            })
        
        # Update posisi partikel
        for particle in self.particles:
            # Interpolasi velocity dari grid
            i = int(particle['x'] / 10 * (self.grid_size - 1))
            j = int(particle['y'] / 6 * (self.grid_size - 1))
            
            i = max(0, min(i, self.grid_size - 1))
            j = max(0, min(j, self.grid_size - 1))
            
            u_interp = self.u[j, i]
            v_interp = self.v[j, i]
            
            # Update posisi
            particle['x'] += u_interp * self.dt * 0.1
            particle['y'] += v_interp * self.dt * 0.1
            particle['age'] += 1
    
    def animate(self, frame):
        """Fungsi animasi untuk matplotlib"""
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Setup plot lagi
        self.ax1.set_xlim(0, 10)
        self.ax1.set_ylim(0, 6)
        self.ax1.set_aspect('equal')
        self.ax1.set_title(f'F1 Airflow Simulation - Frame {frame}', fontsize=14, fontweight='bold')
        self.ax1.set_xlabel('Posisi X (m)')
        self.ax1.set_ylabel('Posisi Y (m)')
        
        self.ax2.set_xlim(0, 10)
        self.ax2.set_ylim(0, 6)
        self.ax2.set_aspect('equal')
        self.ax2.set_title('Pressure Field', fontsize=14, fontweight='bold')
        self.ax2.set_xlabel('Posisi X (m)')
        self.ax2.set_ylabel('Posisi Y (m)')
        
        # Gambar geometri F1
        self.draw_f1_geometry()
        
        # Kalkulasi fields
        self.calculate_velocity_field()
        self.calculate_pressure_field()
        
        # Plot streamlines
        self.ax1.streamplot(self.X, self.Y, self.u, self.v, 
                          density=2, color='cyan', linewidth=1.5, arrowsize=1.5)
        
        # Plot pressure field
        pressure_plot = self.ax2.contourf(self.X, self.Y, self.pressure, levels=20, 
                                         cmap='RdYlBu_r', alpha=0.7)
        
        # Update dan plot partikel
        self.update_particles()
        if self.particles:
            particle_x = [p['x'] for p in self.particles]
            particle_y = [p['y'] for p in self.particles]
            particle_ages = [p['age'] for p in self.particles]
            
            # Warna berdasarkan age
            colors = plt.cm.plasma(np.array(particle_ages) / max(particle_ages) if particle_ages else [1])
            self.ax1.scatter(particle_x, particle_y, c=colors, s=10, alpha=0.7)
        
        # Add colorbar untuk pressure
        if frame == 0:
            plt.colorbar(pressure_plot, ax=self.ax2, label='Pressure (Pa)')
        
        # Add info text
        info_text = f"Wind Speed: {self.wind_speed:.1f} m/s\n"
        info_text += f"Wind Angle: {self.wind_angle:.1f}°\n"
        info_text += f"Reynolds: {self.reynolds_number:.1e}\n"
        info_text += f"Particles: {len(self.particles)}"
        
        self.ax1.text(0.5, 5.5, info_text, fontsize=10, 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def run_simulation(self):
        """Menjalankan simulasi dengan animasi"""
        print("Starting F1 Airflow Simulation...")
        print(f"Grid size: {self.grid_size}x{self.grid_size}")
        print(f"Wind speed: {self.wind_speed} m/s")
        print(f"Wind angle: {self.wind_angle}°")
        print("Press Ctrl+C to stop simulation")
        
        try:
            anim = FuncAnimation(self.fig, self.animate, frames=1000, 
                               interval=100, blit=False, repeat=True)
            plt.tight_layout()
            plt.show()
        except KeyboardInterrupt:
            print("\nSimulation stopped by user")
    
    def save_results(self, filename="f1_airflow_results.json"):
        """Menyimpan hasil simulasi ke file"""
        results = {
            'parameters': {
                'wind_speed': self.wind_speed,
                'wind_angle': self.wind_angle,
                'reynolds_number': self.reynolds_number,
                'air_density': self.air_density,
                'grid_size': self.grid_size
            },
            'velocity_field': {
                'u': self.u.tolist(),
                'v': self.v.tolist(),
                'x': self.x.tolist(),
                'y': self.y.tolist()
            },
            'pressure_field': self.pressure.tolist(),
            'f1_geometry': self.f1_geometry
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def calculate_aerodynamic_forces(self):
        """Menghitung gaya aerodinamis pada mobil F1"""
        # Integrasi pressure di sekitar geometri untuk menghitung gaya
        drag_force = 0
        lift_force = 0
        
        # Simplified calculation
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x_pos = self.X[i, j]
                y_pos = self.Y[i, j]
                
                # Cek apakah point berada di permukaan mobil
                if self.is_on_surface(x_pos, y_pos):
                    # Calculate pressure force
                    pressure_diff = self.pressure[i, j] - 101325
                    
                    # Estimate surface normal (simplified)
                    normal_x, normal_y = self.get_surface_normal(x_pos, y_pos)
                    
                    # Force components
                    force_x = pressure_diff * normal_x * (self.x[1] - self.x[0])
                    force_y = pressure_diff * normal_y * (self.y[1] - self.y[0])
                    
                    drag_force += force_x
                    lift_force += force_y
        
        return drag_force, lift_force
    
    def is_on_surface(self, x, y):
        """Cek apakah point berada di permukaan mobil"""
        # Simplified check
        if 2 <= x <= 8 and 2.5 <= y <= 3.5:
            return True
        return False
    
    def get_surface_normal(self, x, y):
        """Mendapatkan normal vector di permukaan"""
        # Simplified normal calculation
        if 2 <= x <= 8 and 2.5 <= y <= 3.5:
            if abs(y - 2.5) < 0.1:
                return 0, -1  # Bottom surface
            elif abs(y - 3.5) < 0.1:
                return 0, 1   # Top surface
            elif abs(x - 2) < 0.1:
                return -1, 0  # Front surface
            elif abs(x - 8) < 0.1:
                return 1, 0   # Rear surface
        return 0, 0

def main():
    parser = argparse.ArgumentParser(description='F1 Airflow Simulator')
    parser.add_argument('--wind-speed', type=float, default=50.0, help='Wind speed in m/s')
    parser.add_argument('--wind-angle', type=float, default=0.0, help='Wind angle in degrees')
    parser.add_argument('--grid-size', type=int, default=50, help='Grid size for simulation')
    parser.add_argument('--save-results', action='store_true', help='Save results to file')
    
    args = parser.parse_args()
    
    # Create simulator
    simulator = F1AirflowSimulator()
    simulator.wind_speed = args.wind_speed
    simulator.wind_angle = args.wind_angle
    simulator.grid_size = args.grid_size
    
    # Run simulation
    simulator.run_simulation()
    
    # Save results if requested
    if args.save_results:
        simulator.save_results()
        
        # Calculate aerodynamic forces
        drag, lift = simulator.calculate_aerodynamic_forces()
        print(f"\nAerodynamic Forces:")
        print(f"Drag Force: {drag:.2f} N")
        print(f"Lift Force: {lift:.2f} N")
        print(f"Drag Coefficient: {drag / (0.5 * simulator.air_density * simulator.wind_speed**2):.3f}")

if __name__ == "__main__":
    main()