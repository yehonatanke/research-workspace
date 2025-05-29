function [out]=error_diffusion(Image)
[height,width]=size(Image); 
err=zeros(height+1,width+1);   b=zeros(size(Image));     a=double(Image);
for x=1:width,
        for y=1:height,
                if (a(y,x)+err(y,x))<128,
                    b(y,x)=0;
                else
                    b(y,x)=255;
                end;
                diff=(a(y,x)+err(y,x))-b(y,x);
                err(y+1,x)= err(y+1,x)+diff*3/8;
                err(y,x+1)= err(y,x+1)+diff*3/8;
                err(y+1,x+1)= err(y+1,x+1)+diff*1/4;
        end;
end;
out=uint8(b);






