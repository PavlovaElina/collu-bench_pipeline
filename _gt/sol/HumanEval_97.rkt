#lang racket
(define (multiply a b)
  (* (modulo (abs a) 10) (modulo (abs b) 10)))
