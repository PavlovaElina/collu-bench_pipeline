#lang racket
(define (make_a_pile n)
  (for/list ([i (in-range n)]) (+ n (* 2 i))))
