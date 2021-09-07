clear all;
clc;
%inputs of problem
e_min=15;       %earth minor axis
e_maj=20;       %earth major axis
e_init=pi/3;    %earth initial angular position
e_tilt=pi/4;    %tilt in earth's ellipse
m_min=25;       %mars minor axis
m_maj=35;       %mars major axis
m_init=-pi;     %mars initial angular position
m_tilt=pi/10;   %tilt in mars' ellipse
v=0.5;          %velocity of rover



%finding the shortest distance between the planets in one mars year
%min distance
min_day = fminbnd(@(t)distance(e_maj,e_min,e_init,e_tilt,m_maj,m_min,m_init,m_tilt,t,t),0,687);


%finding the time of launch of rover
min_rover_t_e = fminbnd(@(t)(rover_distance(e_maj,e_min,e_init,e_tilt,m_maj,m_min,m_init,m_tilt,v,t)),0,687);
path=@(t_m)(distance(e_maj,e_min,e_init,e_tilt,m_maj,m_min,m_init,m_tilt,min_rover_t_e,t_m)-v*(t_m-min_rover_t_e));
min_rover_t_m = fzero(path,min_rover_t_e);

%plotting the orbits and rover with the above data
X1=earth(e_maj,e_min,min_rover_t_e,e_init,e_tilt);
X2=mars(m_maj,m_min,min_rover_t_m,m_init,m_tilt);
d=distance(e_maj,e_min,e_init,e_tilt,m_maj,m_min,m_init,m_tilt,min_rover_t_e,min_rover_t_e);
rover=@(t)(X1+((t-min_rover_t_e)/(min_rover_t_m-min_rover_t_e))*(X2-X1));
show=0;
figure
hold on
for i = 0:min_rover_t_m
    e_orbit=earth(e_maj,e_min,i,e_init,e_tilt);
    m_orbit=mars(m_maj,m_min,i,m_init,m_tilt);
    plot(e_orbit(1,:),e_orbit(2,:),'bo', 'MarkerSize',5)   
    plot(m_orbit(1,:),m_orbit(2,:),'ro', 'MarkerSize',5)
    if i>=min_rover_t_e && i<=min_rover_t_m
        r=rover(i);
        plot(r(1),r(2),'ko', 'MarkerSize',5)
    end
    axis([-m_maj m_maj -m_maj m_maj])
    pause(0.01)
end
xlabel('x')
ylabel('y')
title('Minimun distance by rover')
%distance vs day plot
figure
days = 0:1:687;
plot(days,distance(e_maj,e_min,e_init,e_tilt,m_maj,m_min,m_init,m_tilt,days,days),'k')
xlabel('day')
ylabel('distance')
title('Distance b/w Earth and Mars vs day')

%visualising minimum path
figure
hold on
days = 0:1:min_day;
e_orbit=earth(e_maj,e_min,days,e_init,e_tilt);
m_orbit=mars(m_maj,m_min,days,m_init,m_tilt);
plot(e_orbit(1,:),e_orbit(2,:),'b-')   
plot(m_orbit(1,:),m_orbit(2,:),'r-')
e_min_pos=earth(e_maj,e_min,min_day,e_init,e_tilt);
m_min_pos=mars(m_maj,m_min,min_day,m_init,m_tilt);
plot([e_min_pos(1) m_min_pos(1)], [e_min_pos(2) m_min_pos(2)],'k-')
legend('Earth','Mars','Minimum')
xlabel('x')
ylabel('y')
title('Minimun distance between the planets')
axis([-m_maj m_maj -m_maj m_maj])

%distance rover needs to travel if launched at a particular day
figure
days = 0:1:687;
plot(days,rover_distance(e_maj,e_min,e_init,e_tilt,m_maj,m_min,m_init,m_tilt,v,days),'k-')
xlabel('day')
ylabel('distance')
title('Distance travelled by rover to intercept Mars')