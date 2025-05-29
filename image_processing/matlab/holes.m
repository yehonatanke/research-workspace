function holes
close all;
A=imread('holes.jpg');
h1=figure;     imshow(A);
set(h1,'position',[68   428   393   267]); title('holes');
B=255-A;   % B is  a negative image of A
L=bwlabel(B,4);
h2=figure;  imagesc(L);    colorbar;   set(h2,'position',[508    69   393   267]);
title('connectivity elements');
cc_num=max(max(L));
cc_size=zeros(1,cc_num);
for j=1:cc_num
    cc_size(j)=sum(sum(L==j));
end
h3=figure;
set(h3,'position',[ 66    71   393   267]);
hist(cc_size,5); title('conncted elements histogram')

cc_tr=find(cc_size>9);
cc_num=numel(cc_tr);  

key=zeros(1,cc_num);
h4=figure;  
set(h4,'position',[510   429   393   267]);
for j=1:cc_num
    TI=(L==cc_tr(j));
    % L is the labels matrix.  cc_tr(j)  is a label.
    imagesc(TI);
    invTI=1-TI;
    imagesc(invTI)
    L1=bwlabel(invTI,4);
    key(j)=max(max(L1));
    pause(1);
end

result=numel(find(key>1)); 
msg=strcat(num2str(result),'   objects  contains  at least 2 holes ');
disp( msg);    msgbox(msg) ;