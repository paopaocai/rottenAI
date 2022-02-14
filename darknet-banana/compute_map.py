from voc_eval import voc_eval
 
rec,prec,ap = voc_eval('results/{}.txt', 'VOCdevkit/VOC2007/Annotations/{}.xml', 'VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'fresh_banana','.')

print('ap',ap)
