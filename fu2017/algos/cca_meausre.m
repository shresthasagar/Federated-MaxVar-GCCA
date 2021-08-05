function score = cca_meausre(Z,G)

[L,K]=size(G);
for i=1:L
    for j=1:L
        T(i,j)=norm(Z(i,:)-Z(j,:),2);
        O(i,j)=norm(G(i,:)-G(j,:),2);
    end
end


score = trace(T'*O)/(sqrt(trace(T'*T))*sqrt(trace(O'*O)));