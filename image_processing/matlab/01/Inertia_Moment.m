function   [m,I]=Inertia_Moment
close all
A=imread('holes.jpg');
B=uint8(255*ones(size(A))-double(A));
L=bwlabel(B,4);
figure(1)
imagesc(L);
colorbar;
set(1,'position',[508    40   418   298]);
impixelinfo;  %pixval(1,'on');
Label=input('enter your label :');  % 15 %
close(1)
[r,c]=find(L==Label);
m=[mean(r)  mean(c)];
M=length(find(L==Label));
S=0;
for k=1:M,
    S=S+(  norm(   [r(k)-m(1)      c(k)-m(2)]   )  )^2;
end
I=S/M;

if (nargout==0)
    disp(strcat(' I=',num2str(I)))
    disp(strcat(' m=',num2str(m)))
    clear I m
end;


