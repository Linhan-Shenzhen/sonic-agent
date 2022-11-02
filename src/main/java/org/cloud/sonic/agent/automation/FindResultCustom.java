package org.cloud.sonic.agent.automation;

import java.io.File;
import lombok.*;

@Getter
@Setter
@ToString
@NoArgsConstructor
public class FindResultCustom {
    private int x;
    private int y;
    private File file;
    private int time;
    private double matchDegree;
}
