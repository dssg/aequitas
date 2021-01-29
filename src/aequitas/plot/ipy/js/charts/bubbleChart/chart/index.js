import React from "react";
import PropTypes from "prop-types";
import { isNull } from "lodash";
// import { Zoom } from "@vx/zoom";

import colors from "~/constants/colors.scss";
import sizes from "~/constants/sizes";

import AxisDisparity from "./AxisDisparity";
import AxisAbsolute from "./AxisAbsolute";
import Row from "./Row";
import ThresholdBands from "~/components/ThresholdBands";

import {
  getScalePositionDisparity,
  getScalePositionAbsolute,
  getScaleSizeBubble
} from "~/utils/scales";

import "./style.scss";

const propTypes = {
  accessibilityMode: PropTypes.bool.isRequired,
  isDisparityChart: PropTypes.bool.isRequired,
  data: PropTypes.array.isRequired,
  fairnessThreshold: PropTypes.number.isRequired,
  handleActiveGroup: PropTypes.func.isRequired,
  metrics: PropTypes.array.isRequired,
  referenceGroup: PropTypes.string.isRequired,
  scaleColor: PropTypes.func.isRequired,
  scaleShape: PropTypes.func.isRequired,
  activeGroup: PropTypes.string
};

function Chart(props) {
  const chartAreaHeight = sizes.ROW_HEIGHT * props.metrics.length;

  const scalePosition = props.isDisparityChart
    ? getScalePositionDisparity(
        props.data,
        props.metrics,
        props.fairnessThreshold
      )
    : getScalePositionAbsolute();

  const scaleBubbleSize = getScaleSizeBubble(props.data);

  return (
    <svg width={sizes.WIDTH} height={chartAreaHeight}>
      {props.isDisparityChart ? (
        <AxisDisparity
          scale={scalePosition}
          chartAreaHeight={chartAreaHeight}
        />
      ) : (
        <AxisAbsolute scale={scalePosition} chartAreaHeight={chartAreaHeight} />
      )}
      {!isNull(props.fairnessThreshold) && props.isDisparityChart ? (
        <ThresholdBands
          fairnessThreshold={props.fairnessThreshold}
          y={sizes.MARGIN.top}
          height={chartAreaHeight}
          scale={scalePosition}
          color={
            props.accessibilityMode ? colors.referenceGrey : colors.thresholdRed
          }
        />
      ) : null}
      {props.metrics.map((metric, index) => {
        const metricAxisY = sizes.ROW_HEIGHT * (index + 0.5);
        return (
          <Row
            key={`row-${metric}`}
            isDisparityChart={props.isDisparityChart}
            metric={metric}
            data={props.data}
            y={metricAxisY}
            referenceGroup={props.referenceGroup}
            scaleColor={props.scaleColor}
            scaleShape={props.scaleShape}
            scalePosition={scalePosition}
            scaleBubbleSize={scaleBubbleSize}
            activeGroup={props.activeGroup}
            handleActiveGroup={props.handleActiveGroup}
          />
        );
      })}
    </svg>
  );
}

Chart.propTypes = propTypes;

export default Chart;
