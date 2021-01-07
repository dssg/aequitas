import React from "react";
import { isNull } from "lodash";
import { scaleLinear, scalePow } from "d3-scale";
import { max } from "d3-array";

import colors from "../colors.scss";
import sizes from "../enums/sizes";
import { highlightColor } from "../utils/colors";

import DisparityAxis from "../components/DisparityAxis";
import Bubble from "../components/Bubble";
import MetricLine from "../components/MetricLine";
import ThresholdBands from "../components/ThresholdBands";

export default function DisparityChart(props) {
  console.log(props.data);
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
          <g height={sizes.ROW_HEIGHT} key={`row-${metric}`}>
            <MetricLine
              y={metricAxisY}
              metric={metric}
              lineStart={sizes.MARGIN.left}
              lineEnd={sizes.WIDTH - sizes.MARGIN.right}
            />
            {props.data.map((group, index) => {
              const groupName = group["attribute_value"];

              let groupColor = props.scaleColor(groupName);

              if (groupName === props.hoverGroup) {
                groupColor = highlightColor(groupColor);
              }

              return (
                <Bubble
                  key={`bubble-${props.metric}-${groupName}`}
                  x={scaleDisparity(group[`${metric}_disparity_scaled`])}
                  y={metricAxisY}
                  size={scaleBubbleSize(group["group_size"])}
                  color={groupColor}
                  index={index}
                  group={group}
                  groupCount={props.data.length}
                  centerShape={props.scaleShape(groupName)}
                  handleHoverGroup={props.handleHoverGroup}
                />
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}
