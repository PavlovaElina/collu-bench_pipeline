#lang racket
(define (sum_to_n n) (for/sum ([i (in-range 1 (+ n 1))]) i))
