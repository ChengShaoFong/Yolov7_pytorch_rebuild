'''
Quantization init parameters : 

    f_quantBitwidth : Bitwidth for feature map quantization
    f_compensateBit : Bitwidth for feature map compensation
    f_compensateScale : Real scale for feature map compensation calculation

    w_quantBitwidth : Bitwidth for weight quantization
    w_compensateBit : Bitwidth for weight compensation
    w_compensateScale : Real scale for feature map compensation calculation
'''

f_quantBitwidth = 8
f_compensateBit = 0
f_compensateScale = 1 << f_compensateBit


# 調整這裡，兩者相加 = 8 
w_quantBitwidth = 8
w_compensateBit = 0
w_compensateScale = 1 << w_compensateBit