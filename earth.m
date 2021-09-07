function x=earth(a,b,t,phi,alpha)
    omega=(1/365)*2*pi;
    x=[cos(alpha) sin(alpha); -sin(alpha) cos(alpha)]*[a*cos(omega*t+phi); b*sin(omega*t+phi)];
end
