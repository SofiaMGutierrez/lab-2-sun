'''
Author: Sofia Gutierrez
Lab #2: The purpose of this lab is to detect and monitor the brightest regions in several images of the sun
'''

import numpy as np
import matplotlib.pyplot as plt
import os

def read_image(imagefile):
    #Reads image in imagefile and returns color and gray-level images
    img = (plt.imread(img_dir+file)*255).astype(int)
    img = img[:,:,:3]  #Remove transparency channel
    img_gl = np.mean(img,axis=2).astype(int)
    return img, img_gl

def plot_sq(I,ax,r,c,h,w):
    p = np.array([[r,c],[r,c+h],[r+w,c+h],[r+w,c],[r,c]])
    ax.plot(p[:,0],p[:,1],linewidth=0.5,color='cyan')

def brightest_pixel(I,ax):
    max_value = np.amax(I) #255
    max_index = np.where(I == max_value)
    #print(max_index)
    index_zip = np.asarray(max_index).T #zip the arrays within max_index
    print(index_zip)
    plot_sq(I,ax,index_zip[0,1]-25,index_zip[0,0]-25,50,50)

def brightest_region_1_1(I,ax,r,c,h,w):
    max_sum = 0
    #focuses on whole image
    for i in range(len(I)-h+1):
        for j in range(len(I[i])-w+1):
            region_sum = 0
            #focuses on square region
            for k in range(i,i+h):
                for q in range(j,j+w):
                    region_sum += I[k][q]
            if region_sum > max_sum:
                max_sum = region_sum
                r, c = k, q
    plot_sq(I,ax,r,c,h,w)

def find_integral_I(I):
    p = I.cumsum(axis=0).cumsum(axis=1)
    p = np.insert(p,[0],0,axis=0)
    p = np.insert(p,[0],0,axis=1)
    return p

def brightest_region_1_2(I,ax,r,c,h,w):
    integral_I = find_integral_I(I)
    max_sum = 0
    for i in range(len(I)-h+1):
        for j in range(len(I[i])-w+1):
            region_sum = 0
            #sends in coordinates of the square/region to integral formula
            region_sum = (integral_I[(i+w),(j+h)])-(integral_I[(i+w),(j)])-(integral_I[(i),(j+h)])+(integral_I[(i),(j)])
            if region_sum > max_sum:
                max_sum = region_sum
                r, c= i, j
    plot_sq(I,ax,r,c,h,w)

def brightest_region_1_2(I,ax,r,c,h,w):
    #focuses on whole image
    for i in range(len(I)-h+1):
        for j in range(len(I[i])-w+1):
            r = I[i:i+h]
            square = r[:,j+w]
    region = np.sum(square)

def brightest_region_2_1(I,ax,r,c,h,w):
    integral_I = find_integral_I(I)
    max_range = [h,w]
    max_value = integral_I[max_range[0],max_range[1]]-integral_I[max_range[0]-h,max_range[1]]-integral_I[h,max_range[1]-w]+integral_I[max_range[0]-h,max_range[1]-w]
    for i in range(h,len(integral_I[0:])):
        for j in range(w,len(integral_I[0,0:])):
            value = integral_I[i,j]-integral_I[i-h,j]-integral_I[i,j-w]+integral_I[i-h,j-w]
            if value > max_value:
                max_value = value
                max_range[0],max_range[1] = i-1, j-1
    plot_sq(I,ax,max_range[1]-w//2,max_range[0]-h//2,h,w)

def brightest_region_2_2(I,ax,r,c,h,w):
    p = np.cumsum(I,1)
    p = np.cumsum(p,0)
    p = np.insert(p,0,0,0)
    p = np.insert(p,0,0,1)
    tl = p[0:len(p)-h,0:len(p[0])-w]
    tr = p[0:len(p)-h,w:len(p[0])]
    bl = p[h:len(p),0:len(p[0])-w]
    br = p[h:len(p),w:len(p[0])]
    
    total = br-tr-bl+tl
    max = np.max(total)
    coord = np.argwhere(total[:] == max)
    r, c = coord[0][1], coord[0][0]
    plot_sq(I,ax,r,c,h,w)

if __name__ == "__main__":  

    img_dir = '/Users/sofiagutierrez/Downloads/lab2imgs/' #Directory where images are stored
    
    img_files = os.listdir(img_dir)  #List of files in directory
    
    plt.close('all')
    
    for file in img_files:
        print(file)
        if file[-4:] == '.png': #File contains an image
            img, img_gl = read_image(img_dir+file)
            fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2)
            ax1.imshow(img)                  #Display color image
            ax2.imshow(img_gl,cmap='gray')   #Display gray-leval image
            plot_sq(img,ax1,0,0,450,450)
            plt.show()
            brightest_pixel(img,ax1)
            plt.show()
            brightest_region_1_1(img_gl,ax2,0,0,5,5)
            plt.show()
            
            brightest_region_1_2(img_gl,ax2,0,0,5,5)
            plt.show()
            brightest_region_2_1(img_gl,ax2,0,0,5,5)
            plt.show()
            brightest_region_2_2(img_gl,ax2,0,0,5,5)
            plt.show()