import cv2
import os
import matplotlib.pyplot as plt
from calibration_utils import calibrate_camera, undistort
from binarization_utils import binarize
from perspective_utils import birdeye
from line_utils import get_fits_by_sliding_windows, draw_back_onto_the_road, Line, get_fits_by_previous_fits
from moviepy.editor import VideoFileClip
import numpy as np
from globals import xm_per_pix, time_window


processed_frames = 0                    # counter of frames processed (when processing video)
line_lt = Line(buffer_len=time_window)  # line on the left of the lane
line_rt = Line(buffer_len=time_window)  # line on the right of the lane


def prepare_out_blend_frame(blend_on_road, img_binary, img_birdeye, img_fit, line_lt, line_rt, offset_meter):
    """
    Prepare the final pretty pretty output blend, given all intermediate pipeline images

    :param blend_on_road: color image of lane blend onto the road
    :param img_binary: thresholded binary image
    :param img_birdeye: bird's eye view of the thresholded binary image
    :param img_fit: bird's eye view with detected lane-lines highlighted
    :param line_lt: detected left lane-line
    :param line_rt: detected right lane-line
    :param offset_meter: offset from the center of the lane
    :return: pretty blend with all images and stuff stitched
    """
    h, w = blend_on_road.shape[:2]

    thumb_ratio = 0.2
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

    off_x, off_y = 20, 15

    # add a gray rectangle to highlight the upper area
    mask = blend_on_road.copy()
    mask = cv2.rectangle(mask, pt1=(0, 0), pt2=(w, thumb_h+2*off_y), color=(0, 0, 0), thickness=cv2.FILLED)
    blend_on_road = cv2.addWeighted(src1=mask, alpha=0.2, src2=blend_on_road, beta=0.8, gamma=0)

    # add thumbnail of binary image
    thumb_binary = cv2.resize(img_binary, dsize=(thumb_w, thumb_h))
    thumb_binary = np.dstack([thumb_binary, thumb_binary, thumb_binary]) * 255
    blend_on_road[off_y:thumb_h+off_y, off_x:off_x+thumb_w, :] = thumb_binary

    # add thumbnail of bird's eye view
    thumb_birdeye = cv2.resize(img_birdeye, dsize=(thumb_w, thumb_h))
    thumb_birdeye = np.dstack([thumb_birdeye, thumb_birdeye, thumb_birdeye]) * 255
    blend_on_road[off_y:thumb_h+off_y, 2*off_x+thumb_w:2*(off_x+thumb_w), :] = thumb_birdeye

    # add thumbnail of bird's eye view (lane-line highlighted)
    thumb_img_fit = cv2.resize(img_fit, dsize=(thumb_w, thumb_h))
    blend_on_road[off_y:thumb_h+off_y, 3*off_x+2*thumb_w:3*(off_x+thumb_w), :] = thumb_img_fit

    # add text (curvature and offset info) on the upper right of the blend
    mean_curvature_meter = np.mean([line_lt.curvature_meter, line_rt.curvature_meter])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blend_on_road, 'Curvature radius: {:.02f}m'.format(mean_curvature_meter), (860, 60), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(blend_on_road, 'Offset from center: {:.02f}m'.format(offset_meter), (860, 130), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return blend_on_road


def compute_offset_from_center(line_lt, line_rt, frame_width):
    """
    Compute offset from center of the inferred lane.
    The offset from the lane center can be computed under the hypothesis that the camera is fixed
    and mounted in the midpoint of the car roof. In this case, we can approximate the car's deviation
    from the lane center as the distance between the center of the image and the midpoint at the bottom
    of the image of the two lane-lines detected.

    :param line_lt: detected left lane-line
    :param line_rt: detected right lane-line
    :param frame_width: width of the undistorted frame
    :return: inferred offset
    """
    if line_lt.detected and line_rt.detected:
        line_lt_bottom = np.mean(line_lt.all_x[line_lt.all_y > 0.95 * line_lt.all_y.max()])
        line_rt_bottom = np.mean(line_rt.all_x[line_rt.all_y > 0.95 * line_rt.all_y.max()])
        lane_width = line_rt_bottom - line_lt_bottom
        midpoint = frame_width / 2
        offset_pix = abs((line_lt_bottom + lane_width / 2) - midpoint)
        offset_meter = xm_per_pix * offset_pix
    else:
        offset_meter = -1

    return offset_meter


def process_pipeline(frame, keep_state=True):
    """
    Apply whole lane detection pipeline to an input color frame.
    :param frame: input color frame
    :param keep_state: if True, lane-line state is conserved (this permits to average results)
    :return: output blend with detected lane overlaid
    """

    global line_lt, line_rt, processed_frames

    # undistort the image using coefficients found in calibration
    img_undistorted = undistort(frame, mtx, dist, verbose=False)

    # binarize the frame s.t. lane lines are highlighted as much as possible
    img_binary = binarize(img_undistorted, verbose=False)

    # compute perspective transform to obtain bird's eye view
    img_birdeye, M, Minv = birdeye(img_binary, verbose=False)

    # fit 2-degree polynomial curve onto lane lines found
    if processed_frames > 0 and keep_state and line_lt.detected and line_rt.detected:
        line_lt, line_rt, img_fit = get_fits_by_previous_fits(img_birdeye, line_lt, line_rt, verbose=False)
    else:
        line_lt, line_rt, img_fit = get_fits_by_sliding_windows(img_birdeye, line_lt, line_rt, n_windows=9, verbose=False)

    # compute offset in meter from center of the lane
    offset_meter = compute_offset_from_center(line_lt, line_rt, frame_width=frame.shape[1])

    # draw the surface enclosed by lane lines back onto the original frame
    blend_on_road = draw_back_onto_the_road(img_undistorted, Minv, line_lt, line_rt, keep_state)

    # stitch on the top of final output images from different steps of the pipeline
    blend_output = prepare_out_blend_frame(blend_on_road, img_binary, img_birdeye, img_fit, line_lt, line_rt, offset_meter)

    processed_frames += 1

    return blend_output


if __name__ == '__main__':

    # first things first: calibrate the camera
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')

    mode = 'images'

    if mode == 'video':

        selector = 'project'
        clip = VideoFileClip('{}_video.mp4'.format(selector)).fl_image(process_pipeline)
        clip.write_videofile('out_{}_{}.mp4'.format(selector, time_window), audio=False)

    else:

        # test_img_dir = 'test_images'
        # for test_img in os.listdir(test_img_dir):

        #     frame = cv2.imread(os.path.join(test_img_dir, test_img))

        #     blend = process_pipeline(frame, keep_state=False)

        #     cv2.imwrite('output_images/{}'.format(test_img), blend)

        #     plt.imshow(cv2.cvtColor(blend, code=cv2.COLOR_BGR2RGB))
        #     plt.show()

        # 동영상 파일 경로 설정 : output_vidio.mp4경로로 변경 => 기존 이미지 말고 동영상에서 작동하는거가 보고 싶어서
        video_path = "C:\\Users\\admin\\sominAI\\self-driving-car\\project_4_advanced_lane_finding\\project_video.mp4"

        # 동영상 프레임 크기 설정 (예: 1920x1080) 
        frame_size = (1920, 1080)

        # OpenCV의 VideoCapture 객체 생성 : OpenCV라이브러리에서 동영상파일 / 카메라로부터 프레임을 캡처하는 데 사용되는 클래스. 
        #동영상 파일이나 실시간으로 들어오는 비디오 스트림으로부터 각 프레임을 읽을수 있으.ㅁ
        cap = cv2.VideoCapture(video_path)

        # OpenCV의 VideoWriter 객체 생성 : 동영상 파일을 생성하는 클래스 
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') #비디오 작성을 위한 코덱을 지정하는 함수. : 코덱 = 데이터를 압축하여 저장하고, 다시 필요할 때 압축을 해제하여 재생하는 역할.
        output_video_path = "C:\\Users\\admin\\sominAI\\self-driving-car\\project_4_advanced_lane_finding\\output_video.mp4" #mp4 비디오 경로
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 25.0, frame_size) # 동영상 파일을 생성하는데 mp4 비디오 경로 + mp4비디오를 코덱으로 한거 + 프레임 속도 + 프레임 크기 

        while cap.isOpened(): #실행하면
            ret, frame = cap.read() #비디오 캡처 객체에서 현재 프레임을 읽어오는 메서드. 여기서 ret은 프레임을 제대로 읽었는가 확인하는것 -> 잘 읽으면 true, 아니면 false, frame = 현재 프레임의 이미지 데이터를 담고 있는 NumPy 배열
            if not ret:
                break

            # 이미지 처리 적용
            blend = process_pipeline(frame, keep_state=True)
            # process_pipline 함수가 비디오의 각프레임을 처리. ture는 연속된 비디오 프레임을 처리할때 유용. false는 현재 프레임을 독립적으로 처리하고 이전의 계산 된 상태를 고려하지 않고 새로운 계산을 하겠다.

            # 동영상 파일로 저장
            video_writer.write(blend)

            # 결과 화면에 표시 (optional)
            cv2.imshow('Result', blend)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # VideoCapture 및 VideoWriter 객체 닫기
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows() # 현재 열려있는 모든 opencv 창을 닫는데 이용.