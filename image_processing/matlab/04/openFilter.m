

txt=double(imread('hebtext.jpg'));
txt=imdilate(255-txt,ones(2));
txt=255-txt;
figure(1)
imshow(uint8(txt));
ind=find(rand(size(Lena))>0.95);
L=double(txt);
L(ind)=0;
figure(2)
imshow(uint8(L))
L1=imerode(255-L,ones(2,2));
figure(3)
imshow(uint8(L1))
L2=imdilate(L1,ones(2,2));
figure(4)
imshow(uint8(L2))
figure(5)
imshow(uint8(255-L2))