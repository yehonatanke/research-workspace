function img1=sharpening(img,alpha,beta)

if (nargin<1)
    
    [FileName,PathName,FilterIndex] = uigetfile('*.jpg','Please select input Image');
    if (FilterIndex==0)
        disp('No input image was selected.');
        return
    end
    
    PicName=strcat(PathName,FileName);
    img=imread(PicName);
    alpha=0.1; %FSPECIAL: ALPHA should be less than or equal 1 and greater than 0.
    ,beta=0.5;
end

h1=zeros(3); h1(2,2)=1;
h = h1-beta*fspecial('laplacian', alpha);
img1= uint8(filter2(h,img,'same'));

if (nargout<1)
    close all
    montage([img img1]);
    clear img
end