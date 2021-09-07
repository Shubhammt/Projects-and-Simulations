function x=distance(a_e,b_e,phi_e,e_tilt,a_m,b_m,phi_m,m_tilt,t1,t2)
    n=length(t1);
    x=zeros(n,1);
    for i=1:n
        x(i)=(sumsqr(earth(a_e,b_e,t1(i),phi_e,e_tilt)-mars(a_m,b_m,t2(i),phi_m,m_tilt)))^0.5;
    end
end