import { scaleOrdinal, scaleLinear, scalePow } from "d3-scale";
import { max } from "d3-array";

import colors from "~/constants/colors.scss";
import shapes from "~/constants/shapes";
import sizes from "~/constants/sizes";

export function getScaleColor(groups) {
  return scaleOrdinal()
    .domain(groups)
    .range([colors.referenceGrey].concat(colors.categoricalPalette.split(",")));
}

export function getScaleShape(groups, accessibilityMode) {
  let shapeRange = [shapes.CROSS].concat(
    Array(groups.length - 1).fill(shapes.CIRCLE)
  );
  if (accessibilityMode) {
    shapeRange = [shapes.CROSS]
      .concat(Array(Math.ceil((groups.length - 1) / 2)).fill(shapes.CIRCLE))
      .concat(Array(Math.floor((groups.length - 1) / 2)).fill(shapes.SQUARE));
  }

  return scaleOrdinal().domain(groups).range(shapeRange);
}

export function getScalePositionDisparity(data, metrics, fairnessThreshold) {
  const maxAbsoluteDisparity = max(
    metrics
      .map((metric) =>
        max(data, (row) => Math.abs(row[`${metric}_disparity_scaled`]))
      )
      .concat(fairnessThreshold)
  );

  return scaleLinear()
    .domain([-maxAbsoluteDisparity, maxAbsoluteDisparity])
    .range([sizes.MARGIN.left, sizes.WIDTH - sizes.MARGIN.right])
    .nice();
}

export function getScalePositionAbsolute() {
  return scaleLinear()
    .domain([0, 1])
    .range([sizes.MARGIN.left, sizes.WIDTH - sizes.MARGIN.right]);
}

export function getScaleSizeBubble(data) {
  const maxGroupSize = max(data, (row) => row["group_size"]);

  return scalePow()
    .exponent(0.5)
    .domain([0, maxGroupSize])
    .range([0, sizes.ROW_HEIGHT / 4]);
}
