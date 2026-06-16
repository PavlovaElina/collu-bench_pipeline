#lang racket
(define (modp n p) (modulo (expt 2 n) p))
