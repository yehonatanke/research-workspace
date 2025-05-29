function holes
% the program  finds the number of objects with  at least 2 holes   in it
%  Input file holes.jpg
clear all;
close all;
A=imread('holes.jpg');
%  reads   holes.jpg into a matrix : A.
figure;
% opens a figure handler and assign it  1
imshow(A)
set(1,'position',[68   428   393   267]);
title('holes');
% B is  a negative image of A
B=uint8(255*ones(size(A))-double(A));
% uint8 cast the matrix element into un-sign integer with 8 bit.
L=bwlabel(B,4);
fhandle=figure;  imagesc(L);    colorbar;   set(fhandle,'position',[508    69   393   267]);
title('connectivity elements');
cc_num=max(max(L));
cc_size=zeros(1,cc_num);

for j=1:cc_num
   % find returns the indices corresponding to the nonzero entries of the
   % array (L==j)
    cc_size(j)=numel(find(L==j));  
end

fhandle=figure;
set(fhandle,'position',[ 66    71   393   267]);
 plot(hist(cc_size,50)); title('#pixel & label')

cc_tr=find(cc_size>9);
cc_num=numel(cc_tr);  

key=zeros(1,cc_num);
fhandle=figure;  
set(fhandle,'position',[510   429   393   267]);
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
disp( strcat(' Number of objects with  at least 2 holes is:',num2str(result)));
