#lang racket
(define (string_sequence n)
  (string-join (map number->string (range 0 (+ n 1))) " "))
