/*
 *  Copyright (C) [SonicCloudOrg] Sonic Project
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */
package org.cloud.sonic.agent.tools.cv;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.opencv.opencv_core.*;
import org.cloud.sonic.agent.automation.FindResultCustom;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class TemMatcherCustom {
    private final Logger logger = LoggerFactory.getLogger(TemMatcherCustom.class);

    public FindResultCustom getTemMatchResult(File temFile, File beforeFile, Boolean isDelete) throws IOException {
        try {
            Mat sourceColor = imread(beforeFile.getAbsolutePath());
            Mat sourceGrey = new Mat(sourceColor.size(), CV_8UC1);
            cvtColor(sourceColor, sourceGrey, COLOR_BGR2GRAY);
            Mat template = imread(temFile.getAbsolutePath(), IMREAD_GRAYSCALE);
            Size size = new Size(sourceGrey.cols() - template.cols() + 1, sourceGrey.rows() - template.rows() + 1);
            Mat result = new Mat(size, CV_32FC1);

            long start = System.currentTimeMillis();
            matchTemplate(sourceGrey, template, result, TM_CCORR_NORMED);
            DoublePointer minVal = new DoublePointer();
            DoublePointer maxVal = new DoublePointer();
            Point min = new Point();
            Point max = new Point();
            minMaxLoc(result, minVal, maxVal, min, max, null);
            rectangle(sourceColor, new Rect(max.x(), max.y(), template.cols(), template.rows()), randColor(), 2, 0, 0);
            FindResultCustom findResult = new FindResultCustom();
            findResult.setTime((int) (System.currentTimeMillis() - start));
            long time = Calendar.getInstance().getTimeInMillis();
            File parent = new File("test-output");
            if (!parent.exists()) {
                parent.mkdirs();
            }
            String fileName = "test-output" + File.separator + time + ".jpg";
            imwrite(fileName, sourceColor);
            findResult.setX(max.x() + template.cols() / 2);
            findResult.setY(max.y() + template.rows() / 2);
            findResult.setFile(new File(fileName));
            return findResult;
        } finally {
            if (isDelete) {
                temFile.delete();
                beforeFile.delete();
            }
        }
    }


    // 增加对目标模板进行比例缩放并匹配，增加匹配可靠性(借鉴HalCon)
    public FindResultCustom getTemMatchResultWithScale(File temFile, File beforeFile, Boolean isDelete) throws IOException {
        this.logger.info("line 81");
        FindResultCustom findResult = new FindResultCustom();
        try {
            this.logger.info("line 84");
            Mat beforeFileMat = imread(beforeFile.getAbsolutePath());
            this.logger.info("line 86");
            int scale_width = beforeFileMat.cols();
            this.logger.info("line 88");
            int scale_height = beforeFileMat.rows();
            this.logger.info("line 90");
            double scale_rate = 1080.0 / scale_width;
            this.logger.info("line 92");
            Mat sourceColor = beforeFileMat.clone();
            this.logger.info("line 94");
            resize(beforeFileMat, sourceColor, new Size(1080, (int) (scale_rate * scale_height)));
            this.logger.info("line 96");
            Mat sourceGrey = new Mat(sourceColor.size(), CV_8UC1);
            this.logger.info("line 98");
            cvtColor(sourceColor, sourceGrey, COLOR_BGR2GRAY);
            this.logger.info("line 100");
            Mat template = imread(temFile.getAbsolutePath(), IMREAD_GRAYSCALE);
            this.logger.info("line 102");
            Size size = new Size(sourceGrey.cols() - template.cols() + 1, sourceGrey.rows() - template.rows() + 1);
            this.logger.info("line 104");
            Mat result = new Mat(size, CV_32FC1);
            this.logger.info("line 106");
            long start = System.currentTimeMillis();
            this.logger.info("line 108");
            List<Mat> scale_tpl_list = scale_pic(template, 0.8, 1.2, 10);
            this.logger.info("line 110");
            List<Double> match_degree_list = new ArrayList<>();
            this.logger.info("line 112");
            List<Point> match_result_list = new ArrayList<>();
            this.logger.info("line 114");
            matchTemplate(sourceGrey, template, result, TM_CCOEFF_NORMED);
            this.logger.info("line 116");
            double[] minVal = new double[1];
            this.logger.info("line 118");
            double[] maxVal = new double[1];
            this.logger.info("line 120");
            Point min = new Point();
            this.logger.info("line 122");
            Point max = new Point();
            this.logger.info("line 124");
            minMaxLoc(result, minVal, maxVal, min, max, null);
            this.logger.info("line 126");
            match_degree_list.add(maxVal[0]);
            this.logger.info("line 128");
            match_result_list.add(max);
            this.logger.info("line 130");
            if (maxVal[0] < 0.9) {
                this.logger.info("line 132");
                for (Mat scale_tpl : scale_tpl_list) {
                    this.logger.info("line 134");
                    matchTemplate(sourceGrey, scale_tpl, result, TM_CCOEFF_NORMED);
                    this.logger.info("line 136");
                    double[] minValTmp = new double[1];
                    this.logger.info("line 138");
                    double[] maxValTmp = new double[1];
                    this.logger.info("line 140");
                    Point minTmp = new Point();
                    this.logger.info("line 142");
                    Point maxTmp = new Point();
                    this.logger.info("line 144");
                    minMaxLoc(result, minValTmp, maxValTmp, minTmp, maxTmp, null);
                    this.logger.info("line 146");
                    match_degree_list.add(maxValTmp[0]);
                    this.logger.info("line 148");
                    match_result_list.add(maxTmp);
                    this.logger.info("line 150");
                }
            }
            Double max_degree = Collections.max(match_degree_list);
            this.logger.info("line 154");
            int max_index = match_degree_list.indexOf(max_degree);
            this.logger.info("line 156");
            Point max_point = match_result_list.get(max_index);
            this.logger.info("line 158");
            Mat max_tpl = scale_tpl_list.get(max_index);
            this.logger.info("line 160");
            rectangle(sourceColor, new Rect(max_point.x(), max_point.y(), max_tpl.cols(), max_tpl.rows()), randColor(), 2, 0, 0);
            this.logger.info("line 162");
            findResult.setTime((int) (System.currentTimeMillis() - start));
            this.logger.info("line 164");
            long time = Calendar.getInstance().getTimeInMillis();
            this.logger.info("line 166");
            File parent = new File("test-output");
            this.logger.info("line 168");
            if (!parent.exists()) {
                this.logger.info("line 170");
                parent.mkdirs();
                this.logger.info("line 172");
            }
            String fileName = "test-output" + File.separator + time + ".jpg";
            this.logger.info("line 175");
            imwrite(fileName, sourceColor);
            this.logger.info("line 177");
            findResult.setX((int) ((max_point.x() + (max_tpl.cols() * 1.0 / 2)) / scale_rate));
            this.logger.info("line 179");
            findResult.setY((int) ((max_point.y() + (max_tpl.rows() * 1.0 / 2)) / scale_rate));
            this.logger.info("line 181");
            findResult.setFile(new File(fileName));
            this.logger.info("line 183");
            findResult.setMatchDegree(max_degree);
            this.logger.info("line 185");
            this.logger.info(findResult.toString());
            this.logger.info("line 187");

        }catch (Exception e){
            e.printStackTrace();
            this.logger.error(e.toString());
        } finally {
            if (isDelete) {
                this.logger.info("line 194");
                temFile.delete();
                this.logger.info("line 196");
                beforeFile.delete();
                this.logger.info("line 198");
            }
        }
        return findResult;
    }


    // 将目标模板按比例缩放，获取缩放后的模板Mat列表
    public static List<Mat> scale_pic(Mat targetMat, double scale_min, double scale_max, int step) {
        System.out.println("line 207");
        if (step < 1 || step >= 100) {
            System.out.println("please input correct para! 1 < step < 100");
        }
        System.out.println("line 211");
        int step_count = 100 / step;
        System.out.println("line 213");
        int tpl_width = targetMat.cols();
        System.out.println("line 215");
        int tpl_height = targetMat.rows();
        System.out.println("line 217");
        List<Mat> tpl_size_list = new ArrayList<>();
        System.out.println("line 219");
        for (int i = 0; i < step; i++) {
            System.out.println("line 221");
            double this_scale = scale_min + i * (scale_max - scale_min) / step_count;
            System.out.println("line 223");
            Mat this_tpl_mat = targetMat.clone();
            System.out.println("line 225");
            resize(targetMat, this_tpl_mat, new Size((int) (this_scale * tpl_width), (int) (this_scale * tpl_height)));
            System.out.println("line 227");
            tpl_size_list.add(this_tpl_mat);
            System.out.println("line 229");
        }
        return tpl_size_list;
    }

    public static Scalar randColor() {
        System.out.println("line 235");
        int b, g, r;
        System.out.println("line 237");
        b = ThreadLocalRandom.current().nextInt(0, 255 + 1);
        System.out.println("line 239");
        g = ThreadLocalRandom.current().nextInt(0, 255 + 1);
        System.out.println("line 241");
        r = ThreadLocalRandom.current().nextInt(0, 255 + 1);
        System.out.println("line 243");
        return new Scalar(b, g, r, 0);
    }
}