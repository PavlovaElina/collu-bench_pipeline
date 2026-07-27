#lang racket
(define (below_threshold l t) (andmap (lambda (x) (< x t)) l))
