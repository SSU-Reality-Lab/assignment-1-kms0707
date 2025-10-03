import sys
import cv2
import numpy as np
import os

def gaussian_blur_kernel_2d(sigma, height, width):
    '''주어진 sigma와 (height x width) 차원에 해당하는 가우시안 블러 커널을
    반환합니다. width와 height는 서로 다를 수 있습니다.

    입력(Input):
        sigma:  가우시안 블러의 반경(정도)을 제어하는 파라미터.
                본 과제에서는 높이와 너비 방향으로 대칭인 원형 가우시안(등방성)을 가정합니다.
        width:  커널의 너비.
        height: 커널의 높이.

    출력(Output):
        (height x width) 크기의 커널을 반환합니다. 이 커널로 이미지를 컨볼브하면
        가우시안 블러가 적용된 결과가 나옵니다.
    '''
    # 좌표 생성
    x = np.arange(-(width // 2), width // 2 + 1)
    y = np.arange(-(height // 2), height // 2 + 1)
    xx,yy = np.meshgrid(x,y)

    # 가우시안 함수 적용하여 커널 계산
    kernel = np.exp(-(xx**2 + yy*2) / (2*sigma**2))

    # 커널의 모든 요소의 합이 1이 되도록 만들기
    kernel_sum = np.sum(kernel)
    if kernel_sum != 0:
        kernel = kernel/ kernel_sum

    return kernel


def cross_correlation_2d(img, kernel):
    '''주어진 커널(크기 m x n )을 사용하여 입력 이미지와의
    2D 상관(cross-correlation)을 계산합니다. 출력은 입력 이미지와 동일한 크기를
    가져야 하며, 이미지 경계 밖의 픽셀은 0이라고 가정합니다. 입력이 RGB 이미지인
    경우, 각 채널에 대해 커널을 별도로 적용해야 합니다.

    입력(Inputs):
        img:    NumPy 배열 형태의 RGB 이미지(height x width x 3) 또는
                그레이스케일 이미지(height x width).
        kernel: 2차원 NumPy 배열(m x n). m과 n은 모두 홀수(서로 같을 필요는 없음).
    '''
    
    '''출력(Output):
        입력 이미지와 동일한 크기(같은 너비, 높이, 채널 수)의 이미지를 반환합니다.
    '''
    kernel_height, kernel_width = kernel.shape

    if img.ndim == 3: #RGB일 경우
        img_height, img_width, img_color = img.shape
    else: #그레이스케일 이미지일 경우
        img_height, img_width = img.shape
        img_color = 1
        img = img.reshape(img_height, img_width, 1)
    # zero 패딩 준비
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    pad_img = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), 'constant', constant_values=0)
    output = np.zeros_like(img)

    # 모든 채널에 대해 반복
    for c in range(img_color):
        #이미지 모든 픽셀에 대해 반복
        for y in range(img_height):
            for x in range(img_width):
                # 영역 자르기
                image_patch = pad_img[y:y + kernel_height, x:x+kernel_width, c]
                output[y,x,c] = np.sum(image_patch*kernel)
    if img_color == 1:
        return output.reshape(img_height, img_width)
    else:
        return output

def convolve_2d(img, kernel):
    '''cross_correlation_2d()를 사용하여 2D 컨볼루션을 수행합니다.

    입력(Inputs):
        img:    NumPy 배열 형태의 RGB 이미지(height x width x 3) 또는
                그레이스케일 이미지(height x width).
        kernel: 2차원 NumPy 배열(m x n). m과 n은 모두 홀수(서로 같을 필요는 없음).

    출력(Output):
        입력 이미지와 동일한 크기(같은 너비, 높이, 채널 수)의 이미지를 반환합니다.
    '''
    kernel_flip = kernel[::-1, ::-1]

    return cross_correlation_2d(img, kernel_flip)

def low_pass(img, sigma, size):
    '''주어진 sigma와 정사각형 커널 크기(size)를 사용해 저역통과(low-pass)
    필터가 적용된 것처럼 이미지를 필터링합니다. 저역통과 필터는 이미지의
    고주파(세밀한 디테일) 성분을 억제합니다.

    출력(Output):
        입력 이미지와 동일한 크기(같은 너비, 높이, 채널 수)의 이미지를 반환합니다.
    '''
    kernel = gaussian_blur_kernel_2d(sigma, size, size)
    return convolve_2d(img, kernel)


def high_pass(img, sigma, size):
    '''주어진 sigma와 정사각형 커널 크기(size)를 사용해 고역통과(high-pass)
    필터가 적용된 것처럼 이미지를 필터링합니다. 고역통과 필터는 이미지의
    저주파(거친 형태) 성분을 억제합니다.

    출력(Output):
        입력 이미지와 동일한 크기(같은 너비, 높이, 채널 수)의 이미지를 반환합니다.
    '''
    low_pass_image = low_pass(img, sigma, size)
    return img - low_pass_image

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)
