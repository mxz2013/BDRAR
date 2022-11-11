import numpy as np
import pandas as pd
import pytz, datetime, cv2, math, os, operator
import matplotlib.pyplot as plt
from skimage import measure
from tqdm import tqdm


def filter_based_on_npy(file_name):
    img_npy = load_npy(mask_dir, file_name)
    good_shadow_pix_ratio = np.sum(img_npy >= score_threshold) / np.sum(img_npy >= 0.01)
    if ratio_threshold[0] <= good_shadow_pix_ratio <= ratio_threshold[1]:
        return round(good_shadow_pix_ratio,3), False  # keep
    else:
        return round(good_shadow_pix_ratio,3), True  # filtered



def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_img_ext(path, img_name):
    possible_ext = ['.jpg', '.png', '.jpeg']
    path = os.path.join(path, img_name)
    for i, ext in enumerate(possible_ext):
        if os.path.exists(path + ext):
            return ext
    print("File extensions are not in the list of \n", possible_ext)

    os.system("ls " + path + "*")
    return None


def add_mask(filename):
    """
    Add mask from the small clipped image to the
    original image.
    SKY: if the small clipped image has A channel (RGBA), use this
    to replace the A channel of the original image.
    """
    orig_ext = check_img_ext(original_dir, filename)
    clip_ext = check_img_ext(clipped_dir, filename)

    image_path = os.path.join(original_dir, filename + orig_ext)
    clipped_image_path = os.path.join(clipped_dir, filename + clip_ext)

    small_clipped_image = cv2.imread(clipped_image_path, -1)
    if small_clipped_image.shape[2] == 3:
        return "NO_MASK"

    original_image = cv2.imread(image_path, -1)  # BGR

    # We resize the small_clipped_image to match the original_image size
    resized_clipped_image = cv2.resize(
        small_clipped_image,
        (original_image.shape[1], original_image.shape[0]),
        interpolation=cv2.INTER_AREA,
    )

    ksize = int(original_image.shape[1] / 500)
    if ksize % 2 == 0:
        ksize = ksize + 1

    resized_mask = resized_clipped_image[:, :, 3]

    media_blur_mask = cv2.medianBlur(resized_mask, ksize)
    _, thresh_mask2 = cv2.threshold(media_blur_mask, 75, 255, cv2.THRESH_BINARY)

    rgb_original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    alpha_original_image = cv2.cvtColor(rgb_original_image, cv2.COLOR_RGB2BGRA)
    alpha_original_image[:, :, 3] = thresh_mask2

    return alpha_original_image  # in BGRA


def get_borders_mask(alpha_original_image, bd_width_perc=0.02):
    """
    Returns the mask of the borders of the image
    """
    max_val = 255
    pixel_n = int(alpha_original_image.shape[1] * bd_width_perc)
    mask = np.zeros((alpha_original_image.shape[0], alpha_original_image.shape[1]))

    mask[:, :pixel_n] = max_val
    mask[:, -pixel_n:] = max_val
    mask[:pixel_n, :] = max_val
    mask[-pixel_n:, :] = max_val
    return mask


def get_clipped_border(alpha_original_image, bd_width_perc=0.10):
    """
    Returns the list of the pixels of the borders of the clipped image.
        alpha_original_image :  BGRA image with alpha as mask
        bd_width_perc : percentage of the width of the border
                        regarding to the width of the image
    """
    mask = alpha_original_image[:, :, 3]
    ks = int(mask.shape[1] * bd_width_perc)
    kernel = np.ones((ks, ks), np.uint8)
    dilation = cv2.dilate(mask, kernel, iterations=1)
    xor_mask = np.logical_xor(mask, dilation)
    return xor_mask


def load_npy(file_path, filename):
    file = np.load(os.path.join(file_path, filename + '.npy')).flatten()
    # with open(os.path.join(file_path, filename + '.npy'), 'rb') as f:
    #     file = np.load(f)
    return file


def check_bg(filename, result_dict, output_path):
    """
    Returns a list containing information about the background
    """
    sd_border_thres = 20
    border_contour_per = 0.20
    result = []

    bgra_original_image = add_mask(filename)

    # SKY use the A channel of the clipped image to replace the A channel of the original image.
    border_mask = get_borders_mask(bgra_original_image)

    # This mask discards the clipped object
    bool_border_clipped_mask = np.logical_and(
        border_mask, np.logical_not(bgra_original_image[:, :, 3])
    )

    # SKY index where border_mask not zero and image = 0 will be true

    intersection_mask = np.logical_and(border_mask, bgra_original_image[:, :, 3])

    n_pixel_intersection_mask = np.sum(intersection_mask)

    aux_per = n_pixel_intersection_mask / np.sum(
        np.logical_and(border_mask, border_mask)
    )

    if aux_per > border_contour_per:
        # object in the original image is too close to the border.
        result.append("OBJECT_TOO_CLOSE")
        result_dict[filename].append("OBJECT_TOO_CLOSE")
        object_close_list.append(1)
        ab_mean_list.append(-1)
        shadow_result_list.append(-1)
        low_sd_list.append(-1)
        bkg_color_list.append(-1)
        shadow_sipp_list.append(-1)
        # client wants to know which images are bad due to object too close
    else:
        object_close_list.append(0)
        var_c1 = math.sqrt(np.var(bgra_original_image[:, :, 0][bool_border_clipped_mask]))
        var_c2 = math.sqrt(np.var(bgra_original_image[:, :, 1][bool_border_clipped_mask]))
        var_c3 = math.sqrt(np.var(bgra_original_image[:, :, 2][bool_border_clipped_mask]))
        border_clipped_sd = (var_c1 + var_c2 + var_c3) / 3

        # First level - Borders and flashy colors
        if border_clipped_sd < sd_border_thres:
            low_sd_list.append(1)
            result.append("LOW_SD_BORDER")
            result_dict[filename].append("LOW_SD_BORDER")
        else:
            low_sd_list.append(0)

        border = bgra_original_image[:, :, :3][bool_border_clipped_mask]
        bkg_colors = [[(0, 0, 255), 75, 0.30, "RED"], [(240, 16, 255), 75, 0.30, "PINK"]]
        with_color = False
        for bkg in bkg_colors:
            border_dist = np.abs(border.astype(np.float32) - bkg[0])
            aux = np.all(border_dist < bkg[1], axis=1)
            counter = len(np.where(aux == True)[0])
            color = counter > int(len(border) * bkg[2])
            if color:
                with_color = True
                result.append(bkg[3])
                result_dict[filename].append(bkg[3])
        if with_color:
            bkg_color_list.append(1)
        else:
            bkg_color_list.append(0)
        # even for HIGH SD BORDER, if a lot of colors are RED or PINK, we still change the BG.

        # Second level - Object contour mask
        # SKY check if we can use the original shadow or synthetic shadow.
        print("len result", len(result))
        if len(result) >= 0:  # low border sd and rich bkg colors
            # SKY Q: how accurate is this way of determine shadow??
            contour_mask = get_clipped_border(bgra_original_image)
            shadow_threshold_area = np.sum(contour_mask)

            hls_orignal_image = cv2.cvtColor(
                bgra_original_image[:, :, :3], cv2.COLOR_BGR2HLS
            )
            region_ksize = 9  # shadow_threshold_area // 3000
            evaluate_shadow(filename, bgra_original_image,
                           region_ksize, ab_mean_list,
                           shadow_result_list, output_path)

def average_entropy(im):
    # Compute normalized histogram -> p(g)
    n_channel = im.shape[-1]
    e = 0
    for i in range(n_channel):
        p = np.array([(im[:, :, i] == v).sum() for v in range(1, 256)])
        p = p / p.sum()
        # Compute e = -sum(p(g)*log2(p(g)))
        e += -(p[p > 0] * np.log2(p[p > 0])).sum()

    return round(e / n_channel,1)


def evaluate_shadow(file_name, org_image: np.ndarray,
                   region_adjustment_kernel_size: int,
                   ab_mean_list: list,
                   shadow_result_list, output_path):
    shadow_ratio, filter_by_npy = filter_based_on_npy(file_name)
    if not filter_by_npy:
        background_original_image = cv2.bitwise_and(org_image[:, :, :3], org_image[:, :, :3],
                                                    mask=cv2.bitwise_not(org_image[:, :, 3]))
        # bgr
        mean_bgr, std = cv2.meanStdDev(background_original_image, mask=cv2.bitwise_not(org_image[:, :, 3]))

        background_mean = int(np.mean(mean_bgr))

        foreground_original_image = cv2.bitwise_and(org_image[:, :, :3], org_image[:, :, :3], mask=org_image[:, :, 3])
        n_objects_foreground = calculate_n_connected_objects(org_image[:, :, 3])
        print("number of foreground objects", n_objects_foreground)

        # lab_background = cv2.cvtColor(img_tmp[:,:,:3], cv2.COLOR_BGR2LAB)
        lab_background = cv2.cvtColor(background_original_image, cv2.COLOR_BGR2LAB)
        hls_background = cv2.cvtColor(background_original_image, cv2.COLOR_BGR2HLS)
        # # entropy_bkg_hls = calculate_entropy(hls_background)
        # print("bkg entropy hls ", entropy_bkg_hls)
        background_entropy = average_entropy(hls_background)
        mean_hls, std_hls = cv2.meanStdDev(lab_background, mask=cv2.bitwise_not(org_image[:, :, 3]))
        background_sd = round(std_hls[0][0],1)

        lab_img_orig = cv2.cvtColor(org_image[:, :, :3], cv2.COLOR_BGR2LAB)

        l_range = (0, 100)
        ab_range = (-128, 127)

        lab_img = lab_img_orig.astype('int16')
        lab_img[:, :, 0] = lab_img[:, :, 0] * l_range[1] / 255
        lab_img[:, :, 1] += ab_range[0]
        lab_img[:, :, 2] += ab_range[0]

        lab_background = lab_background.astype('int16')
        lab_background[:, :, 0] = lab_background[:, :, 0] * l_range[1] / 255
        lab_background[:, :, 1] += ab_range[0]
        lab_background[:, :, 2] += ab_range[0]

        means = [np.mean(lab_img[:, :, i]) for i in range(3)]
        # Calculate the mean values of L, A and B across all pixels
        # Apply threshold using only L
        ab_mean = sum(means[1:])
        ab_mean_list.append(ab_mean)
        fig, axes = plt.subplots(2, 3)
        ax = axes.ravel()
        [axi.set_axis_off() for axi in ax.ravel()]

        # Convert the L,A,B from 0 to 255 to 0 - 100 and -128 - 127 and -128 - 127 respectively
        ax[0].imshow(cv2.cvtColor(lab_img_orig, cv2.COLOR_LAB2RGB))
        ax[0].set_title("Original_"+str(shadow_ratio))

        ax[1].imshow(cv2.cvtColor(foreground_original_image, cv2.COLOR_BGR2RGB))
        ax[1].set_title("foreground_" + str(n_objects_foreground))
        ax[2].imshow(cv2.cvtColor(background_original_image, cv2.COLOR_BGR2RGB))
        ax[2].set_title(str(background_sd) + "_" + str(background_mean) + "_" + str(background_entropy))

        # if background_sd > 30 or background_entropy >= 5.0 or background_mean < 170:
        if background_mean < 100:
            print("filtered bkg")
            shadow_result_list.append(0)
            plt.tight_layout()
            fig_name = file_name + '.jpg'
            fig_no_shadow = os.path.join(output_path, "bad_bkg")
            if not os.path.exists(fig_no_shadow):
                os.makedirs(fig_no_shadow)

            plt.savefig(os.path.join(fig_no_shadow, fig_name))
        else:
            mask_ext = check_img_ext(mask_dir, file_name)
            mask = cv2.imread(os.path.join(mask_dir, file_name + mask_ext), 0)

            kernel = (5, 5)
            mask = cv2.GaussianBlur(mask, kernel, 0)

            _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            kernel_size = (region_adjustment_kernel_size, region_adjustment_kernel_size)
            mask = cv2.erode(mask, kernel_size)
            ax[3].imshow(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))
            ax[3].set_title("deep learning mask")
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            # shadow_mask = np.multiply(mask, background_original_image).astype(np.uint8)
            # shadow_mask = cv2.bitwise_and(mask, mask, mask=cv2.bitwise_not(org_image[:, :, 3]))
            shadow_mask = mask

            labels = measure.label(shadow_mask)

            CHANNEL_MAX = 255
            pix_count = {}
            # Now, we will iterate over each label's pixels

            for label in np.unique(labels):
                if not label == 0:
                    temp_filter = np.zeros(mask.shape, dtype="uint8")
                    temp_filter[labels == label] = CHANNEL_MAX
                    mask_area = mask.shape[0] * mask.shape[1]
                    min_area = mask_area * 0.005
                    max_area = mask_area * 0.3

                    temp_filter = cv2.erode(temp_filter, kernel_size)
                    print("min and max shadow area", min_area, max_area, cv2.countNonZero(temp_filter))
                    if min_area <= cv2.countNonZero(temp_filter) <= max_area:
                        merged_fb_mask = cv2.bitwise_or(temp_filter.reshape(temp_filter.shape[0], temp_filter.shape[1], 1),
                                                        org_image[:, :, 3])

                        n_objects_fb = calculate_n_connected_objects(merged_fb_mask)
                        print("number connected objects in fore and back ground", n_objects_fb)
                        if n_objects_fb == n_objects_foreground:  # shadow is connected to the forground object
                            pix_count[label] = cv2.countNonZero(temp_filter)

            if len(pix_count) == 0:  # no shadow
                shadow_result_list.append(0)
                plt.tight_layout()
                fig_name = file_name + '.jpg'
                fig_no_shadow = os.path.join(output_path, "no_shadow")
                if not os.path.exists(fig_no_shadow):
                    os.makedirs(fig_no_shadow)

                plt.savefig(os.path.join(fig_no_shadow, fig_name))
            else:

                temp_filter = np.zeros(mask.shape, dtype="uint8")
                for key, value in pix_count.items():
                    temp_filter[labels == key] = CHANNEL_MAX
                temp_filter = cv2.erode(temp_filter, kernel_size)
                # final_binary_mask_path = os.path.join(output_path, 'shadow_mask_RGB')
                area = round(cv2.countNonZero(temp_filter) / mask_area, 3)
                contours, hierarchy = cv2.findContours(temp_filter, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                reverse_mask = cv2.cvtColor(cv2.bitwise_not(temp_filter), cv2.COLOR_GRAY2BGR)
                img_w_hole = org_image[:, :, :3] & reverse_mask
                temp_filter = cv2.cvtColor(temp_filter, cv2.COLOR_GRAY2BGR)
                save_path = os.path.join(output_path, 'shadow_mask_RGB')

                fig_name = file_name + '.jpg'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imwrite(os.path.join(save_path, fig_name), temp_filter)

                cv2.drawContours(temp_filter, contours, -1, (255, 0, 0), 3)
                ax[4].imshow(cv2.cvtColor(temp_filter, cv2.COLOR_BGR2RGB))
                ax[4].set_title("area_" + str(area))
                ax[5].imshow(cv2.cvtColor(img_w_hole, cv2.COLOR_BGR2RGB))
                ax[5].set_title("final")

                plt.tight_layout()
                fig_with_shadow = os.path.join(output_path, "with_shadow")
                if not os.path.exists(fig_with_shadow):
                    os.makedirs(fig_with_shadow)
                plt.savefig(os.path.join(fig_with_shadow, fig_name))

                shadow_result_list.append(1)
        if IMSHOW:
            plt.show()

        # Clear the current axes.
        plt.cla()
        # Clear the current figure.
        plt.clf()
        # Closes all the figure windows.
        plt.close('all')


def calculate_n_connected_objects(binary_img, img_show=False):
    thresh = binary_img
    h_max, w_max = thresh.shape[:2]
    h_min = 0.05 * h_max
    w_min = 0.05 * w_max
    area_max = h_max * w_max * 0.9
    area_min = h_max * w_max * 0.005
    output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    # initialize an output mask to store all characters parsed from
    # the license plate
    mask = np.zeros(thresh.shape, dtype="uint8")
    # loop over the number of unique connected component labels
    n_connected = 0
    for i in range(0, numLabels):
        # if this is the first component then we examine the
        # *background* (typically we would just ignore this
        # component in our loop)
        if i == 0:
            text = "examining component {}/{} (background)".format(
                i + 1, numLabels)
        # otherwise, we are examining an actual connected component
        else:
            text = "examining component {}/{}".format(i + 1, numLabels)
        # print a status message update for the current connected
        # component
        print("[INFO] {}".format(text))
        # extract the connected component statistics and centroid for
        # the current label
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # ensure the width, height, and area are all neither too small
        # nor too big
        keepWidth = w > w_min     #and w < w_max
        keepHeight = h > h_min    #and h < h_max
        keepArea = area > area_min # and area < area_max
        # ensure the connected component we are examining passes all
        # three tests
        if all((keepWidth, keepHeight, keepArea)):
            n_connected += 1
            # construct a mask for the current connected component and
            # then take the bitwise OR with the mask
            print("[INFO] keeping connected component '{}'".format(i))
            componentMask = (labels == i).astype("uint8") * 255
            mask = cv2.bitwise_or(mask, componentMask)

            (cX, cY) = centroids[i]
            # clone our original image (so we can draw on it) and then draw
            # a bounding box surrounding the connected component along with
            # a circle corresponding to the centroid
            output = thresh.copy()
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)

            # construct a mask for the current connected component by
            # finding a pixels in the labels array that have the current
            # connected component ID
            componentMask = (labels == i).astype("uint8") * 255
            img_tmp = np.concatenate((output, componentMask), axis=1)
            if img_show:
                cv2.namedWindow("result_" + str(i), cv2.WINDOW_NORMAL)
                cv2.imshow("result_" + str(i), img_tmp)
                key = cv2.waitKey(0)
                if key == ord('n'):
                    cv2.destroyAllWindows()

    return n_connected


if __name__ == "__main__":
    original_dir = "/Users/sky/Documents/data_glovo/original"
    clipped_dir = "/Users/sky/Documents/data_glovo/clipped"
    mask_dir = "/Users/sky/PycharmProjects/BDRAR_SKY/BDRAR/ckpt/BDRAR/2022_11_09_20_50_48/BDRAR_glovo_prediction_3001"
    tz_paris = pytz.timezone('Europe/Paris')
    start_time = datetime.datetime.now(tz_paris).strftime("%Y_%m_%d_%H_%M_%S")
    print("start time", start_time)
    IMSHOW = False
    output_path = "/Users/sky/Documents/data_glovo/select_shadow_after_inference" + start_time
    score_threshold = 0.7
    ratio_threshold = [0.3, 0.7]
    # np.load("/Users/sky/PycharmProjects/BDRAR_SKY/BDRAR/ckpt/BDRAR/calculate_precision_recall/BDRAR_hungerstation_prediction_3001_with_npy/b76a1fe8d460cad0c3cfbea4e7a66c18.npy")

    filename_list = [
        os.path.splitext(f)[0]
        for f in os.listdir(original_dir)
        if os.path.isfile(os.path.join(original_dir, f))
    ]
    # filename_list = ['qqqey3j47enxhmgh8y5v']
    result_dict = {}
    ab_threshold = 100

    img_name_list = []
    ab_mean_list = []
    shadow_result_list = []
    low_sd_list = []
    bkg_color_list = []
    shadow_sipp_list = []
    object_close_list = []

    csv_path = os.path.join(output_path, "analysis.csv")
    n_columns = 0
    for id, file_name in tqdm(enumerate(filename_list[:])):
        orig_ext = check_img_ext(original_dir, file_name)
        clip_ext = check_img_ext(clipped_dir, file_name)
        if orig_ext is not None and clip_ext is not None:
            orig_img_name = file_name + orig_ext
            clip_img_name = file_name + clip_ext
            orig_img_path = os.path.join(original_dir, file_name + orig_ext)
            clip_img_path = os.path.join(clipped_dir, file_name + clip_ext)
            if os.path.exists(orig_img_path) and os.path.exists(clip_img_path):
                clip_img_shape = cv2.imread(clip_img_path, -1).shape
                print("clip shape", clip_img_shape)
                if clip_img_shape[-1] == 4:
                    n_columns += 1
                    print("n_image", n_columns)
                    result_dict[file_name] = []
                    check_bg(file_name, result_dict, output_path)
    df = pd.DataFrame(
        data=zip(img_name_list, ab_mean_list, shadow_result_list, low_sd_list, bkg_color_list, shadow_sipp_list,
                 object_close_list),
        columns=['img_name', 'mean', 'shadow_mask', 'low_border_sd', 'bkg_color', 'sipp_shadow', 'object_too_close'],
        index=None)
    df.to_csv(csv_path)