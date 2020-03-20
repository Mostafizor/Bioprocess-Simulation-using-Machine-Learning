function dC = kinetics(t, C)

% Variable Names
CX = C(1); CN = C(2); CL = C(3);

% Rate Constants
ks = 142.8;
kd = 0.0106; 

% Constant Parameters
Um  = 0.0152; Ud = 8.95e-3; 
KN = 30.0; YNX = 305.0; YLX = 2.304;
Io = 600;

% Mass Balances
dCX = (Um*(Io / (Io + ks))*(CN / (CN + KN))*CX) - (Ud*CX);
dCN = (-YNX)*Um*(CN / (CN + KN))*CX;
dCL = (YLX*Um*(Io / (Io + ks))*(CN / (CN + KN))*CX) - (kd*CL*CX*YLX); 

% Assign output variables
dC(1,:) = dCX;
dC(2,:) = dCN;
dC(3,:) = dCL;


