function x=rover_distance(e_maj,e_min,e_init,e_tilt,m_maj,m_min,m_init,m_tilt,v,t_e)
    n=length(t_e);
    x=zeros(n,1);
    for i=1:n
        path=@(t_m)(distance(e_maj,e_min,e_init,e_tilt,m_maj,m_min,m_init,m_tilt,t_e(i),t_m)-v*(t_m-t_e(i)));
        x(i) = fzero(path,t_e(i));
        x(i) = distance(e_maj,e_min,e_init,e_tilt,m_maj,m_min,m_init,m_tilt,t_e(i),x(i));
    end
end