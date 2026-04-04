# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A single-file Tic Tac Toe game: `tictactoe.html`. Open it directly in any browser — no build step, no dependencies, no server required.

## Architecture

All logic lives in one file:
- **CSS** (inline `<style>`): dark-themed board with pulse animation on winning cells
- **JS** (inline `<script>`): `board` (9-element array), `current` player, `over` flag, `scores` object — game state is entirely in memory and resets on page reload
- Win detection iterates `WINS` (8 hardcoded triplets) after every move
