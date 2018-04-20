#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 预处理

from PIL import Image
import uuid
import numpy as np

def pre(fn, code='aaaa', fp_label=None):
    im = Image.open(fn)
    w,h = im.size
    im = im.convert('L')
    im = im.point(lambda i:0 if i < 120 else 255)
    y_buffer = []
    
    for x in  xrange(im.size[0]):
        s = 0
        for y in xrange(im.size[1]):
            if im.getpixel((x,y)) < 125:
                s += 1
        
        y_buffer.append(s)

    y_sps = []
    b_in = False
    for x in xrange(w):
        if b_in and y_buffer[x] == 0:
            #out
            y_sps.append(x)
            b_in = False
        elif (not b_in) and y_buffer[x] > 0:
            #in
            y_sps.append(x-1)
            b_in = True

    if len(y_sps) == 6:
        #有一个交叠，找到最大的，中分
        max_p = 0
        max_c = -1
        for i in [0,2,4]:
            m = y_sps[i+1] - y_sps[i]
            if m > max_c:
                max_c = m
                max_p = i
        
        md = int((y_sps[max_p+1] - y_sps[max_p]) /2)

        t = []
        t.extend(y_sps[:max_p])
        t.extend([y_sps[max_p], y_sps[max_p] + md, y_sps[max_p] + md+1, y_sps[max_p+1]])
        t.extend(y_sps[max_p+2:])
        y_sps = t

    if len(y_sps) == 8:
        for x in y_sps:
            im.putpixel((x, 1), 125)

        ret = []
        for i, c in zip([0, 2, 4, 6], code):
            x_start = y_sps[i]
            x_end = y_sps[i+1]
            crop = im.crop((x_start, 5, x_end, 45))
            image = Image.new('RGB', (40, 40), (255, 255, 255))
            image = image.convert('L')
            ww = x_end - x_start
            if ww>40:
                print 'error, dig to wide'
                ret = None
                break
            else:
                image.paste(crop, (int((40-ww)/2),0,int((40-ww)/2)+ww,40))
                fn_dig = 'yzm_dig/{}_{}.png'.format(c, uuid.uuid4())
                image.thumbnail((28,28), Image.ANTIALIAS)
                ret.append(np.array(image, dtype=np.float32))
                if fp_label:
                    image.save(fn_dig)
                    fp_label.write('{},{}\n'.format(fn_dig, c))
                    print fn_dig, c
        
        return ret
    else:
        print 'error img, ', fn

    return None



def pre_main():
    with open('yzm/yzm_log.txt', 'r') as fp:
        yzm_logs = fp.readlines()
        print 'we have logs', len(yzm_logs)
        print 'sample', yzm_logs[0]


        with open('test_labels.txt', 'w') as fp_label:
            for i in xrange(8000, len(yzm_logs)):
                line = yzm_logs[i].strip()
                ww = line.split(',')
                if len(ww) != 2:
                    print 'error line', line
                    continue
                fn_img, code = line.split(',')
                code = code.upper()
                print '>>>>', i, fn_img, code

                # 检查code的合法性
                if len(code) == 4:
                    is_code_ok = True
                    for c in code.upper():
                        c = ord(c)
                        if (c>=ord('A') and c<=ord('Z')) or (c>=ord('0') and c<=ord('9')):
                            pass
                        else:
                            print 'illegal char', c
                            is_code_ok = False
                            break

                    if is_code_ok:
                        pre(fn_img, code, fp_label)

def pre_to_mem(fn_yzm):
    ims = pre(fn_yzm)
    if ims:
        pix = np.array(ims, dtype=np.float32)
        pix = pix * np.array([1.0/255])
        pix = np.reshape(pix, (4, 28*28))
        pix = pix.astype(np.float32)
        print(pix.shape, pix[:1,])
        return pix
    else:
        return None