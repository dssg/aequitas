import React from "react";
import PropTypes from "prop-types";
import { isNull } from "lodash";
import { scaleLinear, scalePow } from "d3-scale";
import { max } from "d3-array";

import colors from "../colors.scss";
import sizes from "../enums/sizes";

import DisparityAxis from "../components/DisparityAxis";
import ThresholdBands from "../components/ThresholdBands";
import DisparityRow from "./DisparityRow";

const propTypes = {
  accessibilityMode: PropTypes.bool.isRequired,
  data: PropTypes.array.isRequired,
  fairnessThreshold: PropTypes.number.isRequired,
  handleActiveGroup: PropTypes.func.isRequired,
  metrics: PropTypes.array.isRequired,
  referenceGroup: PropTypes.string.isRequired,
  scaleColor: PropTypes.func.isRequired,
  scaleShape: PropTypes.func.isRequired,
  activeGroup: PropTypes.string,
};

function DisparityChart(props) {
  const chartAreaHeight = sizes.ROW_HEIGHT * props.metrics.length;
  const fullHeight = chartAreaHeight;
  sizes.MARGIN.top + sizes.MARGIN.bottom;

  const maxAbsoluteDisparity = max(
    props.metrics.map((metric) =>
      max(props.data, (row) => Math.abs(row[`${metric}_disparity_scaled`]))
    )
  );

  const maxGroupSize = max(props.data, (row) => row["group_size"]);

  const scaleDisparity = scaleLinear()
    .domain([-maxAbsoluteDisparity, maxAbsoluteDisparity])
    .range([sizes.MARGIN.left, sizes.WIDTH - sizes.MARGIN.right])
    .nice();

  const scaleBubbleSize = scalePow()
    .exponent(0.5)
    .domain([0, maxGroupSize])
    .range([0, sizes.ROW_HEIGHT / 4]);

  return (
    <svg width={sizes.WIDTH} height={fullHeight}>
      <DisparityAxis scale={scaleDisparity} chartAreaHeight={chartAreaHeight} />
      {!isNull(props.fairnessThreshold) ? (
        <ThresholdBands
          fairnessThreshold={props.fairnessThreshold}
          y={sizes.MARGIN.top}
          height={chartAreaHeight}
          scale={scaleDisparity}
          color={
            props.accessibilityMode ? colors.referenceGrey : colors.thresholdRed
          }
        />
      ) : null}
      {props.metrics.map((metric, index) => {
        const metricAxisY = sizes.ROW_HEIGHT * (index + 0.5);
        return (
          <DisparityRow
            key={`row-${metric}`}
            metric={metric}
            data={props.data}
            y={metricAxisY}
            referenceGroup={props.referenceGroup}
            scaleColor={props.scaleColor}
            scaleShape={props.scaleShape}
            scaleDisparity={scaleDisparity}
            scaleBubbleSize={scaleBubbleSize}
            activeGroup={props.activeGroup}
            handleActiveGroup={props.handleActiveGroup}
          />
        );
      })}
    </svg>
  );
}

DisparityChart.propTypes = propTypes;
export default DisparityChart;
