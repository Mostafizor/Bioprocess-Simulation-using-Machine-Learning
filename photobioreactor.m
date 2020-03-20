% Define initial concentrations
C0 = [0.2, 800, 0];

% Define length span
tspan = [0, 300];

% run ODE solver
[t, y] = ode45(@kinetics, tspan, C0);

t0 = 0:12:300;
y0 = interp1(t, y, t0);