import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define trainable final time
class TrainableTime(nn.Module):
    def __init__(self, initial_t_f):
        super().__init__()
        self.t_f = nn.Parameter(torch.tensor(initial_t_f, dtype=torch.float32))

    def forward(self):
        return self.t_f


# Define the neural network (PINN)
class FNN(nn.Module):
    def __init__(self, N_input, N_output, N_hidden, N_hidden_nodes, activation_fn):
        super().__init__()
        layers = [nn.Linear(N_input, N_hidden_nodes), activation_fn]
        for _ in range(N_hidden - 1):
            layers.extend([nn.Linear(N_hidden_nodes, N_hidden_nodes), activation_fn])
        layers.append(nn.Linear(N_hidden_nodes, N_output))
        self.model = nn.Sequential(*layers)

    def forward(self, t):
        output = self.model(t)
        y1, y2, y3, y4 = output[:, 0:1], output[:, 1:2], output[:, 2:3], output[:, 3:4]
        lambda1, lambda2, lambda3, lambda4 = (
            output[:, 4:5],
            output[:, 5:6],
            output[:, 6:7],
            output[:, 7:8],
        )
        return y1, y2, y3, y4, lambda1, lambda2, lambda3, lambda4


# Training function
def train_model(activation_fn, N_input, N_output, N_hidden, N_hidden_nodes, iterations, initial_t_f):
    torch.manual_seed(42)
    pinn = FNN(N_input, N_output, N_hidden, N_hidden_nodes, activation_fn)
    t_f_param = TrainableTime(initial_t_f)

    # Initial optimizer: Adam
    optimizer_adam = torch.optim.Adam(list(pinn.parameters()) + [t_f_param.t_f], lr=1e-3)

    # LBFGS optimizer
    optimizer_lbfgs = torch.optim.LBFGS(
        list(pinn.parameters()) + [t_f_param.t_f], 
        max_iter=50000, 
        history_size=50, 
        lr=1.0, 
        tolerance_grad=1e-7, 
        tolerance_change=1e-9, 
        line_search_fn="strong_wolfe"
    )

    # Boundary and domain points
    t_domain = torch.linspace(0, 1, t_dm_pts).view(-1, 1).requires_grad_(True)
    t_bc = torch.tensor([[0.0]], requires_grad=True)
    t_final = torch.tensor([[1.0]], requires_grad=True)  # Scaled final time

    # Loss weights
    w_bc, w_domain, w_term = 0.9, 0.1, 0.45

    # List to store loss history
    loss_history = []
    tf_history = []

    # Store states and control at every 100 iterations
    states_history = []

    for i in range(iterations):
        # Switch optimizer after 70% of iterations
        if i >= int(0.7 * iterations):
            optimizer = optimizer_lbfgs
        else:
            optimizer = optimizer_adam

        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()

            # Boundary losses
            y1_bc, y2_bc, y3_bc, y4_bc, _, _, _, _ = pinn(t_bc)
            loss_BC1 = (
                (y1_bc - 0) ** 2
                + (y2_bc - 0) ** 2
                + (y3_bc - 0) ** 2
                + (y4_bc - 0) ** 2
            )

            t_final_scaled = t_final * t_f_param.t_f
            # Terminal constraint
            _, y2_final, y3_final, y4_final, lambda1_final, lambda2_final, lambda3_final, lambda4_final = pinn(t_final_scaled)
            terminal_constraint = (
                1
                + lambda2_final * y4_final
                + a * lambda3_final * torch.cos(lambda4_final / lambda3_final)
                + a * lambda4_final * torch.sin(lambda4_final / lambda3_final)
            )
            loss_terminal = terminal_constraint ** 2

            loss_BC2 = (
                (y2_final - 5) ** 2
                + (y3_final - 45) ** 2
                + (y4_final - 0) ** 2
                + (lambda1_final - 0) ** 2
            )
            loss_BC = loss_BC1 + loss_BC2

            # Domain losses
            t_scaled = t_domain * t_f_param.t_f
            y1, y2, y3, y4, lambda1, lambda2, lambda3, lambda4 = pinn(t_scaled)

            dy1_dt = torch.autograd.grad(y1, t_scaled, torch.ones_like(y1), create_graph=True)[0]
            dy2_dt = torch.autograd.grad(y2, t_scaled, torch.ones_like(y2), create_graph=True)[0]
            dy3_dt = torch.autograd.grad(y3, t_scaled, torch.ones_like(y3), create_graph=True)[0]
            dy4_dt = torch.autograd.grad(y4, t_scaled, torch.ones_like(y4), create_graph=True)[0]
            dlambda1_dt = torch.autograd.grad(lambda1, t_scaled, torch.ones_like(lambda1), create_graph=True)[0]
            dlambda2_dt = torch.autograd.grad(lambda2, t_scaled, torch.ones_like(lambda2), create_graph=True)[0]
            dlambda3_dt = torch.autograd.grad(lambda3, t_scaled, torch.ones_like(lambda3), create_graph=True)[0]
            dlambda4_dt = torch.autograd.grad(lambda4, t_scaled, torch.ones_like(lambda4), create_graph=True)[0]

            # State equations
            loss_states = (
                (dy1_dt - y3) ** 2
                + (dy2_dt - y4) ** 2
                + (dy3_dt - a * torch.cos(lambda4 / lambda3)) ** 2
                + (dy4_dt - a * torch.sin(lambda4 / lambda3)) ** 2
            )

            # Costate equations
            loss_costates = (
                (dlambda1_dt - 0) ** 2
                + (dlambda2_dt - 0) ** 2
                + (dlambda3_dt + lambda1) ** 2
                + (dlambda4_dt + lambda2) ** 2
            )

            # Final loss
            total_loss = w_bc * loss_BC.mean() + w_domain * (loss_states.mean() + loss_costates.mean())
            if total_loss.requires_grad:
                total_loss.backward()
            return total_loss

        if isinstance(optimizer, torch.optim.LBFGS):
            optimizer.step(closure)
        else:
            optimizer.zero_grad()
            closure()
            optimizer.step()

        # Append loss and t_f to history
        loss_history.append(closure().item())
        tf_history.append(t_f_param.t_f.item())

        # Save states and control at every 100 iterations
        if i % 100 == 0:
            t_scaled = t_domain * t_f_param.t_f
            y1, y2, y3, y4, lambda1, lambda2, lambda3, lambda4 = pinn(t_scaled)
            control = torch.atan2(lambda4, lambda3)  # Updated control calculation
            states_history.append({
                "iteration": i,
                "t": t_scaled.detach().numpy().flatten(),
                "y1": y1.detach().numpy().flatten(),
                "y2": y2.detach().numpy().flatten(),
                "y3": y3.detach().numpy().flatten(),
                "y4": y4.detach().numpy().flatten(),
                "control": control.detach().numpy().flatten(),  # Updated control
            })

            # Save t_f for every 100 iterations
            tf_data = {
                "iteration": i,
                "t_f": t_f_param.t_f.item()
            }
            tf_df = pd.DataFrame([tf_data])
            tf_df.to_csv("final_time_history.csv", mode='a', header=not i, index=False)

    return pinn, t_f_param, loss_history, tf_history, t_domain, states_history


# Load analytical solution
analytical_data = pd.read_csv("analytical.csv")
t_analytical = analytical_data["time"].values
y1_analytical = analytical_data["x_opt"].values
y2_analytical = analytical_data["y_opt"].values
y3_analytical = analytical_data["u_opt"].values
y4_analytical = analytical_data["v_opt"].values
control_analytical = analytical_data["control_u"].values

# Extract analytical final time
analytical_final_time = t_analytical[-1]  # Last time point in the analytical data


# Animation function for states
def animate_states(states_history, t_analytical, y1_analytical, y2_analytical, y3_analytical, y4_analytical):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()

    def update(frame):
        for ax in axs:
            ax.clear()

        # Plot PINN predictions
        t_pinn = states_history[frame]["t"]
        y1_pinn = states_history[frame]["y1"]
        y2_pinn = states_history[frame]["y2"]
        y3_pinn = states_history[frame]["y3"]
        y4_pinn = states_history[frame]["y4"]

        axs[0].plot(t_pinn, y1_pinn, label="PINN y1", color="blue")
        axs[0].plot(t_analytical, y1_analytical, label="Analytical y1", linestyle="--", color="red")
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("y1")
        axs[0].legend()

        axs[1].plot(t_pinn, y2_pinn, label="PINN y2", color="blue")
        axs[1].plot(t_analytical, y2_analytical, label="Analytical y2", linestyle="--", color="red")
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("y2")
        axs[1].legend()

        axs[2].plot(t_pinn, y3_pinn, label="PINN y3", color="blue")
        axs[2].plot(t_analytical, y3_analytical, label="Analytical y3", linestyle="--", color="red")
        axs[2].set_xlabel("Time")
        axs[2].set_ylabel("y3")
        axs[2].legend()

        axs[3].plot(t_pinn, y4_pinn, label="PINN y4", color="blue")
        axs[3].plot(t_analytical, y4_analytical, label="Analytical y4", linestyle="--", color="red")
        axs[3].set_xlabel("Time")
        axs[3].set_ylabel("y4")
        axs[3].legend()

        plt.suptitle(f"Iteration: {states_history[frame]['iteration']}")

    ani = animation.FuncAnimation(fig, update, frames=len(states_history), interval=200)
    ani.save("state_evolution.gif", writer="imagemagick")
    plt.show()


# Animation function for control
def animate_control(states_history, t_analytical, control_analytical):
    fig, ax = plt.subplots(figsize=(10, 6))

    def update(frame):
        ax.clear()
        t_pinn = states_history[frame]["t"]
        control_pinn = states_history[frame]["control"]

        # Plot PINN control
        ax.plot(t_pinn, control_pinn * 180 / np.pi, label=f"PINN Control (Iteration {states_history[frame]['iteration']})", color="blue")

        # Plot analytical control
        ax.plot(t_analytical, control_analytical, label="Analytical Control", linestyle="--", color="red")

        ax.set_xlabel("Time")
        ax.set_ylabel("Control")
        ax.set_title(f"Control Evolution (Iteration {states_history[frame]['iteration']})")
        ax.legend()
        ax.grid(True)

    ani = animation.FuncAnimation(fig, update, frames=len(states_history), interval=200)
    ani.save("control_evolution.gif", writer="imagemagick")
    plt.show()


# Configuration
N_input = 1
N_output = 8  # 4 states + 4 costates
N_hidden = 5
N_hidden_nodes = 40
t_dm_pts = 100
iterations = 10000
a = 100.0  # Parameter 'a' in the equations
initial_t_f = 0.7  # Initial guess for t_f
activation_name = "Tanh" 

# Train the model
activation_fn = "Tanh"
print(f"Training with activation: {activation_name}")
pinn, t_f_param, loss_history, tf_history, t_domain, states_history = train_model(
    activation_fn=activation_fn,
    N_input=N_input,
    N_output=N_output,
    N_hidden=N_hidden,
    N_hidden_nodes=N_hidden_nodes,
    iterations=iterations,
    initial_t_f=initial_t_f,
)

# Load and plot final time history
# This should be done AFTER training, as the file is created during training.
# After training, load and plot final time history
tf_df = pd.read_csv("final_time_history.csv")
plt.figure(figsize=(10, 6))
plt.plot(tf_df["iteration"], tf_df["t_f"], label="PINN t_f", color="blue")
plt.axhline(y=analytical_final_time, label="Analytical t_f", linestyle="--", color="red")
plt.xlabel("Iteration")
plt.ylabel("Final Time (t_f)")
plt.title("Evolution of Final Time (t_f)")
plt.legend()
plt.grid(True)
plt.savefig("final_time_evolution.png")
plt.show()  # Ensure the plot is displayed

# Animate the results
animate_states(states_history, t_analytical, y1_analytical, y2_analytical, y3_analytical, y4_analytical)
animate_control(states_history, t_analytical, control_analytical)