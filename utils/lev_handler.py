import os
import re
import string
from typing import List, Literal
import numpy as np

class TableCell(object):
    EditTypes = Literal['del', 'ins', 'match', 'sub', 'first']

    def __init__(self, cost: int = 0, trace: EditTypes = 'first'):
        super().__init__()
        self.cost, self.trace = cost, trace


def build_dp(n_ref: int, n_hyp: int) -> List[List[TableCell]]:
    grid: List[List[TableCell]] = []

    for i in range(n_ref + 1):
        if i == 0:
            row = [TableCell(cost=j, trace='ins') for j in range(n_hyp + 1)]
        else:
            row = [TableCell() for _ in range(n_hyp + 1)]
            row[0] = TableCell(cost=i, trace='del')
        grid.append(row)

    grid[0][0] = TableCell(cost=0, trace='first')

    assert isinstance(grid, list) and all(
        isinstance(rw, list) and all(isinstance(c, TableCell) for c in rw) for rw in grid)
    return grid


def fill_cell(i: int, j: int, grid: List[List[TableCell]], ref_tokens: List[str], hyp_tokens: List[str]) -> None:
    choices = []

    same = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
    diag_op = 'match' if same == 0 else 'sub'

    if i > 0:
        choices.append((grid[i - 1][j].cost + 1, 'del'))
    if j > 0:
        choices.append((grid[i][j - 1].cost + 1, 'ins'))
    if i > 0 and j > 0:
        choices.append((grid[i - 1][j - 1].cost + same, diag_op))

    choices = sorted(choices, key=lambda x: x[0])
    best_cost, best_op = choices[0]

    k = 1
    precedence = ['match', 'sub', 'ins', 'del']
    while k < len(choices) and choices[k][0] == best_cost:
        if precedence.index(choices[k][1]) < precedence.index(best_op):
            best_op = precedence[precedence.index(choices[k][1])]
        k += 1

    grid[i][j] = TableCell(cost=best_cost, trace=best_op)
    return


def walk_back(grid, i, j):
    cur = grid[i][j].cost

    if i == 0:
        return i, j - 1
    if j == 0:
        return i - 1, j

    if grid[i - 1][j - 1].cost < cur:
        return i - 1, j - 1
    if grid[i][j - 1].cost < cur:
        return i, j - 1
    if grid[i - 1][j].cost < cur:
        return i - 1, j

    if grid[i - 1][j - 1].cost == cur:
        return i - 1, j - 1
    if grid[i][j - 1].cost == cur:
        return i, j - 1
    if grid[i - 1][j].cost == cur:
        return i - 1, j

    raise ValueError('THIS SHOULD NEVER HAPPEN')


def summarize(grid):
    n_ref = len(grid) - 1
    n_hyp = len(grid[0]) - 1

    if n_ref <= 0 or n_hyp <= 0:
        error_rate = float('inf')
    else:
        error_rate = grid[n_ref][n_hyp].cost / n_ref

    n_del = 0
    n_sub = 0
    n_ins = 0

    i, j = n_ref, n_hyp
    ops = []

    while i > 0 or j > 0:
        op = grid[i][j].trace
        ops.append(op)
        if op == 'sub':
            n_sub += 1
        elif op == 'del':
            n_del += 1
        elif op == 'ins':
            n_ins += 1
        i, j = walk_back(grid, i, j)

    return (error_rate, ops, n_del, n_sub, n_ins)


def Levenshtein(r, h):
    n_ref, n_hyp = len(r), len(h)

    grid = build_dp(n_ref, n_hyp)

    for i in range(1, n_ref + 1):
        for j in range(1, n_hyp + 1):
            fill_cell(i, j, grid, r, h)

    return summarize(grid)


def logistic_lev_path(ref, cand):
    _, lev_path, _, _, _ = Levenshtein(ref, cand)
    logistic_path = []
    for op in lev_path:
        if op == 'match':
            logistic_path.append('high')
        elif op == 'sub' or op == 'ins':
            logistic_path.append('low')

    return logistic_path


def multiclass_lev_path(ref, cand):
    _, lev_path, _, _, _ = Levenshtein(ref, cand)
    return lev_path


def complete_lev_path(ref, cand):
    _, lev_path, _, _, _ = Levenshtein(ref, cand)
    return lev_path