import tinycolor from "tinycolor2";

export function highlightColor(color) {
  return tinycolor(color).darken(10).toString();
}
