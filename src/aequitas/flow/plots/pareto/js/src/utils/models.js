/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

import { isNull } from "lodash";

export const id = (model) => model["model_id"];
export const isPareto = (model) => model["is_pareto"];

export const sortModels = (models, recommendedModel, pinnedModel, selectedModel) => {
    /* We sort the models so that model points are drawn in the correct order:
     "recommended" is drawn after "pareto" which are drawn after "non-pareto" points. */
    const sortedModels = models.slice().sort((a, b) => {
        if (id(a) === id(recommendedModel)) return 1;
        if (id(b) === id(recommendedModel)) return -1;

        if (id(a) === id(pinnedModel)) return 1;
        if (id(b) === id(pinnedModel)) return -1;

        if (!isNull(selectedModel)) {
            if (id(a) === id(selectedModel)) return 1;
            if (id(b) === id(selectedModel)) return -1;
        }

        if (isPareto(a)) return 1;
        if (isPareto(b)) return -1;

        return 0;
    });

    return sortedModels;
};
